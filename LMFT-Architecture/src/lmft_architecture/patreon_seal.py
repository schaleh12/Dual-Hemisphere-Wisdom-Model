"""Anti-Boolean-Flipper Cryptographic Seal (`patreon_seal.py`).

Plain-English Purpose:
Prevents anyone who clones the public GitHub repository from bypassing the Patreon password
lock simply by editing `def is_patreon_unlocked(): return True` in Python source code.

Instead of relying on a boolean `if` check guarding plaintext code, the core Bloodmoon
execution methods (`_sealed_run_phase_1_female_producer`, `_sealed_run_phase_2_male_audience`,
`_sealed_run_phase_3_rem_sleep`, `_sealed_seed_cycle_with_custom_script`,
`_sealed_handle_chat_message`, and `_sealed_handle_microwave_step`) are removed from
`bloodmoon_app.py` and stored only as authenticated `PBKDF2-HMAC-SHA256` (`200,000` rounds) +
`HMAC-SHA256-CTR` ciphertext inside `bloodmoon_patreon_seal.enc`.

Without the 512-bit `seal_key_hex` (which can only be mathematically derived from the
Patreon password or read from the local git-ignored `patreon_unlock_receipt.json` after
a one-time unlock), the ciphertext cannot be decrypted or executed in memory.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any
import zlib

DEFAULT_SEAL_PBKDF2_ITERATIONS = 200_000
SEAL_DOMAIN_SUFFIX = b":bloodmoon-seal-v1"

SEALED_BLOODMOON_CORE_BUNDLE: dict[str, str] = {
    "schema": "lmft_patreon_seal_v1",
    "cipher": "HMAC-SHA256-CTR+ZLIB",
    "nonce_hex": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
    "ciphertext_hex": (
        "43188692f3e519d1801474fc9a470a83ce7a9ed143bae3a7d3426812ff8f177932223e5d9b0fe50d67b4e31f94"
        "7a63af4e7043effe377a7870f4b53b62880da82093bf9a952a543fcd1deab6565812086fcef293336c7d9a9e03"
        "9402932484b2cfe42add0ff9249e3b89d5f62526a68e44de6a76a55417b35965ccdf63b67eb94f78f4a611d0e8"
        "6c3fc1f00745774d3999d6409813443cd5718628116eba211f796c44c02afa01cb4a95c246108bf9c431ac980a"
        "00c8691170f5485f7c5d8c2956595435ec8a047caab7e8924a0e378873b86c30422fbb1ab7affd063e0f160424"
        "281352807300ff3514bcd3859050d7e29bfa7bbbdf8860660f9de8a3e5370cf55e67f8a384fde2dd9164d0f302"
        "d7a4c4a1d12a431f8f6374e242be0e1304a69332abfa8de6c83ad17d117fd5a265f0c5cc63ad540f18a44e4f63"
        "05f704068b981ebddbb3db40a16bbcb8f0ae7653c5294877573a1df4b7e864e52353f23eb5d16c24815ae1603b"
        "5e506b15945d189bf37233d3d2cbc7912deeefbcb125ead25debd279c3fef3aac31d5df9bdbb02d50cb4f55d3c"
        "9af41fd09f94bf649a934ed7eade84ee148c4ed237c2477423f17f75b87d5cf731621937b4eaa4dc7a3d7fd732"
        "8705740d9ff01f2b820c361091df3b7c2b3162e9a461696f6c0133c7984b825221f9c023907c8a31be4f4485e9"
        "a2ff17d3d0812414cc8a6538b17486ec97b19ce6ba4cf4a4ad20c031d1cffa04bdabae057324048622baf88686"
        "c484c1a2e13b0cb78b6741e7d6a2e901decde007b1338b845538cdc4733895e6672ee42c7807a2a459aa79edec"
        "3ed2fb3bdd20ccdb9976fa1a320e5ac100eee04d1f77ee1a66b40436794f9fb3d515f8787b20e27c02b95ab4e8"
        "19f841baff243ddb7e21076d13b9a8c52e1feac739f0a0181f97c6aa48831ccc97ff3c5a04b4db544625ddedf4"
        "1b5b59ca254b77dc8ebf92ca03df30575cfd971f7fb35707ce566bc5cd5ca18c5e682f5563bea26d4b63d72621"
        "92ddd331d271d0d0ab3f354aaab4c62593e6bd6a477f2cd53d80783432fc36ea64234d943070affed230e9d7ab"
        "1dce491aa465bdf7cc33769552e14ef31fe51659983fdb6835eab95a7d68cf72c85c1bf36103abb13892296a26"
        "788da463107c0b1a127d5bbf22c021c6b0b227bf9e1e99ee675691e31b25e47d47259b59a9f49ae9a2d592be3a"
        "e9b2c405b443b1ea66bb802c4c80c32da8fbe21b0c2e9291cec5dfe77e97180ff80cae796f0c439ce1a0c6b089"
        "335986a82e2f8e2c0620d61795ed9afbe018a6b10525370e43e167358a07bbae69643aa9804b43cb43c8207b45"
        "472b06d69c4286bdc110bc22043ce4b800f945582f4937c2df846d210b775f60929425a7524d37fc9d52ef0205"
        "0a0954694295f03f44d3848df0859e28e5f38b50dcfdc84de70c8b1de28eae0108774e3188db8bcbe67d4bda42"
        "b6b983dbfa4e4a8745a9306d0f4ebc12fe4c78552b661a9acd55276a24948e5f187971075d0011385aa91c59cd"
        "e12264ad8fb2b8e5a9422027856359798f2ad9bedebfed92a31b0e749039d4482afc81a8b13b7c61900e091f53"
        "9e7a5f1e93cd2e1328af8760f917e97e6b0580a68fba1828ed4d919a520619c4a35bc7559d4813be54e27d5edc"
        "211d20f51b7336062d335c4e67e8ec447659c3386f1837d8f18d6b727fd0ff64dbf9c81bc7031475ffcbb324f4"
        "c74e3cbc3bcf0e3a9c156f54bf9294315a973263f82cbe2a307ddaae484987bd121d9219c8910d2ea9a61c7378"
        "5d22e423f4c66929d8dcb7407812ed2190b19660f446d87c1c68c55f2c5f58786c4751777c1c9ca1edc5b38b97"
        "3c75ed8ba927944e918b9077c66b531d99df4e845f04555c02a0e9e6b4f9835dc1c01ad661aea4a219bd20155b"
        "00879f7d5612723e55d3c07e7c70838d9aa30f19c4a832c2a8621cc0f8480f46680e2f06e446edd8fbba8d1e7c"
        "e30688241f0b1b7be1261e79ad2407f97d668c527b1308c4d30bc1587e626b8f41dd1d0f9eff17e80526785e7a"
        "cdd8b0d1341e5724c4da44b9efdf10bedc6bc7e063c64769226e28fb3d2a1ae76f718cbff1fef2374d268603a6"
        "0252ddcdc4e04824df47affefdb2f72e98b0492f1736e37282220254c1f7f0a1bd2bfa0b218be1c077357a41a6"
        "cde8805a9e68eab3d573d186929cdf9dcfe07a89ff769b19f43fd961290dbe3c9745deb75967ddf87ae288e61a"
        "9dec2fb55c655e39779c29106be7cd5d1e385f8e71f7ada40ec7ce1e5addc82ac07be83600973b8b123c85cc53"
        "b538864a38a075bb1140eb0204e6a5361e4df4b077692c08f9ff45fcfcd57a09a6680ce062baefbe9c78b80323"
        "c9950bfd770a89d834e1d5640e712cbb17cc477cd1667edc96b4d3518e5ff4797421d0e775d0fae32d354b0b95"
        "e45e8d03e3b8f6065586b0f4222b52cdd2d1552862dd1bc976a4a72051b631a279b8cbe7bd9dc3e6014f13fc3f"
        "81e8312af2d60aeefda58bfcef43e9e56af433bb1b9d23ab40daa8571a1fb1ae577f1f2f2d21c4ba7c92118460"
        "c8e7bda2ea3b7e97a57b47232f0c88a0c737a054f2136a1b258744a776190683c0f7619f7b0811beffbf2c3131"
        "a9e4909e8fb3c28b2a64d907403e262a9f541edeb977b597e6b12baca8a60450e69b9e4dd7440a996ca27bc0ab"
        "c2dbc49fbcf605da5bb9a82d4b02e7a99b40bd86c2ebc0f4d0c57d774b21d89df8be5480cf5be84bc416bec518"
        "6401f193f3c3715d3f7699970b907c5900f9be8b162b79928f6ac427f78a91b05f85e76c495653f7056489d624"
        "331f77e71b3dac6127b8cf51dc192434d49d75b8512d8f1febe98ab468ce7d43db06e068423ed593741fe9a1dc"
        "43a299d104d7fa6db94df928576a68a34aa88da91f0af592e8b89d48f17299998dda10fe4d6148c6e7a05a469c"
        "a785157a9ee0c101b1635dec59a8134f5ff56239f62b8c6fc698d66a55091f70897b3f59e24ae49588f3db402d"
        "3baffe20a85b7835dcd268eb896fa47669d0b082287028b893436a6b6d35f4f31b38cb9ad8caafe6b25921363a"
        "a9cb46119a511ad363bc2c091545392e896e95bffd1d6ab445daa9671392cd4dba93cca1837cfb1659c9e72315"
        "d463033ebe0794ba068332734fba084d68a985327125231e16867dd994e33e1f771c9cc1d3fab4957672666ee5"
        "94074dac10926cb819ec6e1805c671ef4efac8452bc12ba219105725278c9b2c9aba923f3eb31c53f0dcb7c958"
        "c72fc7fa27e4edfaa2e8132a696b7601ca7f409999ae311483cb51feab3199e5cbabf0dff88290b830747bd142"
        "2ba0d29689299fdda6d694527c400e6d6b34d9f7a8ad08254a11c2e8d9501d45e9081abf3bb1bf3b8091f95c9b"
        "7d26088f0fc4e7d1ccf0019df8cc6de3e913352faeb07f48ede5a7c42426a3b29fad774761dd75eb7c9cfcce1e"
        "bb08322946644f939866c7a2d66bf72622d24edf6878f725d5de8cef4054b5af4566be1ee3bf41b27229c7ee63"
        "a12eca73ff146581ee47c09f9bfbe191a91f5a2753f951fd3fe49cf80a82aef4292afa3084dd1beb0c0af65e5a"
        "70d0150851a7253d7461f5a9f34cb7f27a6b7bb97f95a21597e603aea088e4e60adf1e08d591f35b8343f07c68"
        "cf72dc287463e49158aa9909fb3d2964a2a3aec4e30768dc66549fec64104f30c706ab01b7a2bc2b23dfeb79a3"
        "902c6c185c836719fc8fa6664319a4b39c18f06f807c3051887ed656769805c09712d800ca6bb1f22de8d008c1"
        "bd4b2673ca36e5515de3ae657d1267aead5579a9cfb23e4a4592366e6498e49078047d552b829d910e1a40bf00"
        "f41dd5fd14720e9a8e95adb3da1cf8f1097215e64a85a70376e528993e5ad7d4b064763ef25c7aa142d8c58314"
        "2d485b1ae02f004fb3626f15694312de47b516b13f606dbe814897a15236d9d3c29c9279954fb33154371bf00b"
        "da2a6a23b46cd88fab4447805ee33f678c1b19ae1e47f02677719695ca49ce7f514485f5dafb41ee44295cffb6"
        "54554e9973da5dea099ba0dd16fd9e91fbd58740d05221eef91b97d103d9ce80a6715f10c69a5298d4b05b94bf"
        "3b24b8f8601dc3274b52346a31449a78cccf346edbe81ddc610f2ed60a668062ed5668a2fc910618168fe460c4"
        "8c656aa62312c164085f4ff095b454ea6e8153c47d3e58c236eafe4e7ec768efb658a611e228dc8ae2c8ddcac7"
        "29456ba45a8c5edd783ebe80ea574ab29bb08c8c2222a75d8c3c53017ae6a9d8fc83fd985bd04ab92d61d36dfd"
        "ba670a19e6d47cf6edab33ed9f1d3c020bd17258326e9f553816e3d2768b860774a5ba5443a00a551ee82d1712"
        "36791beeac090d1d1ffec6d7b1b9f8ee8335beca4b10b963f6976cd0fbf2ebec854930bde80af1d58f0288706e"
        "a5a9df936f6640d96130732000d42f9168783d50d386cf6c94d2f9a71abb5493c22dd28b43c43b56632277aa11"
        "d165cbdd4c86d9e0449c46162759c3502197bf1f4e99c2e21f21ea8fc8275d4492d2ec5073f2698775375add49"
        "94660609f4759153e68defd1767042df9cc9af30fd2006c6e6c8d760e825dd4e5d2a078c5f52e6ee78e112eecf"
        "e180fab4e76aa976e7446a39ca9df93816ea50c63b628fbd37119745e65f38ffbbf87a9eedb9901e66d5908abb"
        "0b2754d5314a9bae8bcddb567ec44269548327721f95e4d4c59330c4c79d9305d11e2b69a0ce9c827dfbecfe9d"
        "4379f05be03a0a3ad8b2f1cf235f5ed10f714659ad763e952032334b3c821acee7e5273f56e0f544c6b4e0179c"
        "31d95c72e0685d934590dfa4eb97b5673eb507e45cec4d601e17609befab4d36378cceba829c47728a9403f642"
        "725dc4847c19fcf79e220f5de6b3495567cf977b71a4f06d229962b5822b88c01fdf3b345f07c4622d9e367896"
        "65161f58d30e7add11d759c5bced037bd89d320cf61c61a8008009ee23b9d835b427425f9e942c0f18ab7825b8"
        "65b4c5f5523dcab6f616135c2dfa3aba68efb3653c9dfabe53fe1421e4259c2974139b902b5b53394e35e7ae21"
        "2b52d5f5e22ad620f41cd4bf9443f2cd2a89f0072b5b5ef88a496e120daedfb628f075b86c44d88844d291732e"
        "1f11ba2547d6bbc0de5ef123859803d661b5cd76b4da1297157258e47bed85c81abfe1ded9dd2eb0761da05f1b"
        "4752e6964f5f0f4d27ab388458bdeb887f0a59433161a3129587b11f2f0d16319376ce010773505ea24e8e80ec"
        "fb841990cc68c04c6d0f5f82be14cd475d480e7b62a1f21442fb812d104bdd73d3e0a8454c1ffcaf232dcf777d"
        "12cabcb20ff786f343976a986a2c0543737b791b515c106ba1eaa935b4636ecaa6c779367220eb4efff44fb47e"
        "5365368c5916fb7988b164b03221febcb7816261338a8c8ed7e0a6d6ed9db06717a2abfbcad6d67e775226cc0d"
        "12cb58cc5e15c5013ab464a198ee9dee3498140ced8a715f6bf0b3e1f834d6cc67f6e1878120faa5ee8ba87c58"
        "eb829d40d8619a0de42ffcf702e3ef907454a8f34e419702061d5e6661cf443567d25f45b387186d8097b1aa18"
        "b37c65f743f33cf4e19cbb3b3cd4b111636c6fd28e46efd43d263b4d13ec5298415e575fb927915fca81d838dc"
        "b8498b37a133e03058bbe804dbff827b6ba08ad7a492baaa0f92066bbf390b0c697910a34a28bcef4a25cf7269"
        "a72c0970c19e7e71eba4b37ce8d1f18f45d023d47f4df1cc404468095650b93483d43371a233fd22c8b28c57fb"
        "7c14a9a8c45092ee9dd108885b93d0f84242f007b662cc7bdc1b52a600fc99f0de83c6b03e05f19036166ee3bc"
        "f185e764bd5f094e2a93b3892b082861b278b2d3a84fa9c429062f275a7c69acf1f6eab7cb98c6ebc6ab703308"
        "fc3db59c3a08ed1b373e79aad1bf4962321194ed26e37010baf302629f5e56d360d81c764442"
    ),
    "hmac_sha256": "fe0cc941c61100f33da70fb5236f7d40102c6619dbbb7a2d7c94720ac5ec0e71",
}

def _keystream_hmac_ctr(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    """Generate a deterministic cryptographic keystream using HMAC-SHA256 in counter mode."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(
            enc_key,
            nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _xor_bytes(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def derive_patreon_seal_keys(
    passphrase: str,
    *,
    salt_hex: str,
    iterations: int = DEFAULT_SEAL_PBKDF2_ITERATIONS,
) -> tuple[str, str]:
    """Derive `(verify_digest_hex, seal_key_hex)` from `passphrase` and `salt_hex`.

    - `verify_digest_hex` (32 bytes / 64 hex chars): one-way verification digest.
    - `seal_key_hex` (64 bytes / 128 hex chars): 32-byte `enc_key` + 32-byte `mac_key`
      under domain separation (`salt_bytes + b":bloodmoon-seal-v1"`), which cannot be
      computed from `verify_digest_hex`.
    """
    clean = passphrase.strip()
    if not clean:
        raise ValueError("Passphrase cannot be empty.")
    salt_bytes = bytes.fromhex(salt_hex)
    verify_digest_hex = hashlib.pbkdf2_hmac(
        "sha256",
        clean.encode("utf-8"),
        salt_bytes,
        int(iterations),
        dklen=32,
    ).hex()
    seal_key_hex = hashlib.pbkdf2_hmac(
        "sha256",
        clean.encode("utf-8"),
        salt_bytes + SEAL_DOMAIN_SUFFIX,
        int(iterations),
        dklen=64,
    ).hex()
    return verify_digest_hex, seal_key_hex


def encrypt_sealed_python_module(
    source_code: str,
    seal_key_hex: str,
    *,
    nonce_bytes: bytes | None = None,
) -> dict[str, str]:
    """Compress and encrypt Python source code with `HMAC-SHA256-CTR` + `HMAC-SHA256` authentication tag."""
    key_raw = bytes.fromhex(seal_key_hex.strip())
    if len(key_raw) != 64:
        raise ValueError("seal_key_hex must be a 64-byte (128-hex-char) derived key.")
    enc_key = key_raw[:32]
    mac_key = key_raw[32:]
    nonce = nonce_bytes if (nonce_bytes is not None and len(nonce_bytes) == 16) else os.urandom(16)

    compressed = zlib.compress(source_code.encode("utf-8"), level=9)
    stream = _keystream_hmac_ctr(enc_key, nonce, len(compressed))
    ciphertext = _xor_bytes(compressed, stream)
    tag_hex = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).hexdigest()

    return {
        "schema": "lmft_patreon_seal_v1",
        "cipher": "HMAC-SHA256-CTR+ZLIB",
        "nonce_hex": nonce.hex(),
        "ciphertext_hex": ciphertext.hex(),
        "hmac_sha256": tag_hex,
    }


def decrypt_and_load_sealed_module(
    seal_bundle: Mapping[str, Any] | str | Path | None = None,
    seal_key_hex: str | None = None,
    *,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate, decrypt, and load the sealed Bloodmoon core namespace in memory.

    Raises `PermissionError` if `seal_key_hex` is missing, malformed, or fails the
    `HMAC-SHA256` authentication check—ensuring boolean-flipping in `bloodmoon_app.py`
    can never unlock or execute the sealed ciphertext.
    """
    if not seal_key_hex or not str(seal_key_hex).strip():
        raise PermissionError(
            "Cryptographic seal locked: valid Patreon decryption key required to decrypt "
            "the sealed Bloodmoon core — flipping booleans cannot decrypt ciphertext."
        )
    try:
        key_raw = bytes.fromhex(str(seal_key_hex).strip())
    except ValueError as exc:
        raise PermissionError("Cryptographic seal locked: malformed seal_key_hex.") from exc

    if len(key_raw) != 64:
        raise PermissionError(
            "Cryptographic seal locked: 512-bit Patreon seal key required to decrypt Bloodmoon core."
        )

    if seal_bundle is None:
        raw_bundle: Mapping[str, Any] = SEALED_BLOODMOON_CORE_BUNDLE
    elif isinstance(seal_bundle, Mapping):
        raw_bundle = seal_bundle
    else:
        path = Path(seal_bundle)
        if not path.exists():
            raise PermissionError(f"Cryptographic seal file missing: {path}")
        raw_bundle = json.loads(path.read_text(encoding="utf-8"))

    nonce = bytes.fromhex(str(raw_bundle["nonce_hex"]))
    ciphertext = bytes.fromhex(str(raw_bundle["ciphertext_hex"]))
    expected_tag = str(raw_bundle["hmac_sha256"]).strip().lower()

    enc_key = key_raw[:32]
    mac_key = key_raw[32:]
    computed_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).hexdigest().lower()
    if not hmac.compare_digest(computed_tag, expected_tag):
        raise PermissionError(
            "Cryptographic seal locked: HMAC-SHA256 authentication failed for bloodmoon_patreon_seal.enc "
            "(invalid Patreon key; boolean flipping cannot bypass ciphertext encryption)."
        )

    stream = _keystream_hmac_ctr(enc_key, nonce, len(ciphertext))
    compressed = _xor_bytes(ciphertext, stream)
    try:
        source_code = zlib.decompress(compressed).decode("utf-8")
    except Exception as exc:
        raise PermissionError("Cryptographic seal locked: decompression failed.") from exc

    ns: dict[str, Any] = {"__builtins__": __builtins__}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source_code, "<bloodmoon_patreon_seal.enc>", "exec"), ns)  # noqa: S102
    return ns
