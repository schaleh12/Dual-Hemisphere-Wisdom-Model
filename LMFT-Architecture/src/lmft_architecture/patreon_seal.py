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
        "43188695f1e521d1841434fceac52a8cce7a36f5a024a5cdc0ba0575b21d0dfa7878818c38bf7979"
        "168802092b873b404dffdb282234428a6e99da35d35dc9647f39e02c2b62795bfa85365e369bcb9ca"
        "595db493dc05cd5ee305c5fa1ea7c62688bb3fe2c7373558fc14c8c2cb14d3603b247acd31b6f3225"
        "da1da78566e882f0e3a6680471bf4712e7de3f994f18f5682e89f58764960c2cb546385bf3851b691"
        "c460cd6cbe9fabdce5c29c40998484a4ed6f97b3f5f9ea4551212bc31dd82bda16120b82066870091"
        "4583ec0fc118b67a035753f2409a1ececc391eba80118849cb06c3f3bc62d55a6dc83de4c3f624ff9"
        "0971587ac489e6de07f667d8d854648290c824613ed808bd51a6b06212d47cba5312e3a0d530ef37b"
        "3962fb1e52673d7c76621914bde43d3da21e919c3941d625f09bc6ce7447ac36ce158de721d996e69"
        "d3d3cca3c79633205b7561bcf96bfd30d1bba129045b25ac31f091f15e51495e4aa5f33bbb973c3d6"
        "92ab7e9ea0391a9a7d9b62fec944715f0cb1e60de9794f27fb43f92c60aa0e499fec3d563ea454f0e"
        "5fe7650297f85344a84a2061b3024aeea39d71ddda293b09173200e37bbe6d29e15384ef6825cf230"
        "483b60ea046807118264af4122e84db6c6562d38415ad3819d9f43e60f77803c429d4aefe5b4de437"
        "129d4d3746115b979f709ee7fa43fd0e5bdc008636b105a95e4f52858b16bb6ef365afc06f0ec7b3e"
        "8855945885390d31690e1546f9fb4c18ac4a5bee9bc87b8118aa67623648438aa3b2eee3588713526"
        "600f65736c9125b5600c0cc85cb224ec617704a5a847d02f0ee48bc3cc199280fedaa349392094850"
        "8724f0d3309a2e77a80d5376d3c8d5e9a4a187a852729bcb7ea3eaf3a7fcb5918c6006aea4e4c5b20"
        "ba12fedbe49a403dc32f840cc7b761df7c4602adfbf5c95577307c99deb259a5c84d3d5714a09bb27"
        "40edd4f5ba45df19c91e25b0ad221b67df68d6c197883e71d53bff717346be3f7035b544b46a89acc"
        "40659f01fd87bfae542598f103691981c36119860589af6f4454d0ea2795b25160fdf1ee5c2b31724"
        "d08aa104a394d107ff6d3ca8124f7861615814c1d855129de4f5d9de87787b2e9196caa8fe3c7bca2"
        "6b9bf02a182993ca8acc6e665b9e54d2cb0b974ec72e35f726e1a12296ef62a9ae55b004f3d09371d"
        "4e22b17845d38d4e321e901203d96b7d94d9d3e64e13ca9c38c5848b07f8ec12f902fdf3cfee601fb"
        "f6957538386e808955fd609e79c84e827a44e94d6723aafb050275283fb39876edb924c90d475e6d9"
        "b123432eeaabd59ef955b715230c86f2893f141b7d24c39c3c5885a3e10c11714e66b48764fef9079"
        "eb39bcc894088843d987c13e24420b5999c39a67deef2e6c15b79395c3b55994f5d562ae17a34e7f3"
        "339990706621f004f717e0de3c3a831a994be2f50bd48b433876f4196cdfeeb6e550faa54df7017c3"
        "7f49aa9ab90c1df8cb033df23318bd3941f3037b96bb63acbd3774ae086f30a7dc88d3ca129af07ea"
        "37e224e06b931aa453ecb94d4d933565b2e15002890ce583ed53dfd2b39c9fe8643a6cf0eecfacf8f"
        "151320c006ec54e79a730f8822976367b6ed18ba88015674092862026b4fb59375973d71dbdf192bf"
        "b1a76d07f6acc429abeeb06d0bd72ea388735d84ff225aca5530b725349c6f9572cf2de602f0f08c0"
        "12568505cee4514e4debdd36b248f16ed6bc467515956c445682237f0690036c4fcd00878d1f8ef9f"
        "8d079c1adebf871d1115dd71882e0f59ca6db3374a3f9abe3e4e9e8b46fe6994c2eec035682e98124"
        "770227a4c3d68d1ca255a3b685cd52ac65dd1719f8c2cebe5dd26959a4dbfeecf7626f5059b3f7746"
        "bbe3d1c28727b819dc8e968ca8610af61192a143a48ac236c4fd60a21eea31d7ce86ab5d62354fd73"
        "f600c5bc00cbc946a1eccf7776c7ddb862b3720539ccc59b63d1dfd79a994f67d3c2987d450b18cc4"
        "80fc4b0467b8c34f4dbdcd180f2e57193da68becfb54b91e7809f1c09e60a83b3f77dce02e39d5ed7"
        "de49175302fdc4b730bdda4e76389c6c487edfa645ab091ad911b6606d59af083183c239664c28c43"
        "6027fbc0afb8d0fccbdb19dd2b4eecb44c874174c14d0bd2f3d988bd30e8e74a9ce07bd00b0a8dd47"
        "5d3cea9060d73d1ccc962bea3be3611d8144e302e0bb0faae5f4a545cf985f2018da0c3c9dfe3ee53"
        "1d529612a7af15115700c7387461b77986847d70a6f4b7731cbc7346d7195899b603b0989c894c222"
        "b40025a8bba33c639bc12062aedf212812e92ac27413664fb9d68e4b50dc3ee0e9114a3284c1f0fde"
        "1896490f84f98d0be18219ff2eb54951bff3fedc65ceecedb1f0b77aa7e2133c4905927df9f79f709"
        "c7f79ebfcd602abf00acc3c19f5fb559ea47b0fa564b5fed764d78c336eca3e812d59dd744a3c8860"
        "acb6be2ed781fd49d89d544f18a7783504bd2e783fb90240dad2a296d6de1187efa8940bdec2e6db6"
        "b1a8a324ae8eb9e79a18ea827e712b05a02f7ba2f0f288340ade374d5a9b8119b839606ecb75b3b9f"
        "5772c05b368f68e714528c9a066d411b94ceb1290e95e0ca9dbb472c7e05fac493531698ec404b2b9"
        "c41b9fdd311dc55832e75cdf3326ca6e6a81b579a27be6c3909f0e76c8e2dfa831e67bc9aaa03bfdb"
        "306e8edaffd7b4712bd1054e109342ea72d8bd2bd331eb3a6bc3633303c6626348f71793396045fc6"
        "8a96c70bfc287cb1ab40ca2b2d626033189ba398cc21bb6a9f585456016c873b45d91a1286e4e9c4e"
        "6d53001ec51a63c6b07145b61172b4b18673b8ce013ca588ab19b2cb4e7689899a8ca14736ca72f7a"
        "f4c5da60734827e301896bcc75366090cd479aea12713472a29bc493c21911115121b612993933009"
        "dfd7326ca4674aba46852216304a83c667a9d55c738b74acb63190e5d6b8a09e77dcdb2d46911b4cf"
        "ed28dd415a883d08255332d32bd2b55e60004fbd57c4ad3971471f319f65154871f220f92627876ba"
        "17787acaa30b85af754147787d21ebf8acf00f6ec07ab2f0f667e6f35a3cffdf33574fd887a000d03"
        "9a76396dcea7148cac1062f4382b62dbbbc2e4ae7c2e956f93b659fcb519ff6ce98d742b666f4ad39"
        "b114d8e4132ab8a6304d426dfe4f515de9d2ef14fe850fe198aad8a46c7cf5dafbf6fda3c9e38b165"
        "40f5652825430f736cf82908c70f1bca4b68cc4dfbc043704af11ba00776d78d3acfd387f09d165c5"
        "ccc5e1bc24a675e1092a342d7eecc3dccd249f11a0eb38a1f4d4ecb4e196f03be01689e9687e13137"
        "38b755117bd45d637a123f5f1c50d4ef58e05c10ec7cfe2ff5db9946294cf591f586d4b3e7d03a106"
        "e24a012c6f4451c79e0296bcb4b682e993422eb6e9cc2bf9c5bf1832d7189aa95dafe83c283155b77"
        "5eb1197e77d5bdd51a9c63625b88b8fa41e339471d2157e8df704f040392cbadf970cc314f8c19cc2"
        "c815c02b5e4f92b7d77b14cc1af265000d5fa768e62e2113bdbc74f08e2c45e92779545abd60ceeb3"
        "d6b88acc63bc52a26198abed6585a97bb24ba0aee82c82ac9367d1fe5eb6de0d0d216e5076848e5d2"
        "c001eefe5693beeb485a2f5b250a16fc49e329f937960ae8d6766ecb0f54efeaf3c278af4bea26243"
        "6d6c895ce6050ba90771b23d009e509ba1859b9dfa678bc344cf60a944ae7aa6fa1a2def789d7d5bf"
        "85d3d3012d8c1bfed821ce4ff156b1d15abdd617da3c4a3ebe060f8002b4702fa997b66beea11bd27"
        "0d7c1057aed16f2df441cb127588ab3836f74440c3c30be74b1c9d39b8cc3d530664a722589501280"
        "5a6497afdb00d2318b08abbb9859387b1b11755dafcb39f67b922ba5200b22d4baedcd80c9a640cca"
        "b6597c9c54132f24e56c04b428fff0603183a2209efde4b31fee1cec33f5e7f3d2a4b0543aad187d8"
        "0067e64326d47fa04ff3437823e0e1a092312799d218f1459f4ed9bb22f619406f25dfdc039decfa2"
        "f11846cb398708dcd3421bd24dfd483bf6438527453be959db8e82a4d1169c60b5767f6a477e07c62"
        "1cfab9a244159ec356a4340438c1cec3260a504fc39b9a5480d7a7e9ece4df36b75a68e37cb64f267"
        "99f26889c38da3b99d786a27e84d33f90bcf4a119afb5fef030d53a3132e1b91fa30897eb0f8183da"
        "99a8814f5cbf66f01f0893c0e6791ca47039a0fe47bd791bfce99fc6dca5d0b0d64492696f2f8be63"
        "f9f7ed25401bc281724fc2ef164c938b03120f582ef1ee66f2e700af7bdbbc85b03c16ff6e8a47ba3"
        "e8fa546ce3b1a20005b4895414a5500e2ddff98ed940c1eedc6a67c3d7bb67b1a94df3fe827200de5"
        "5d5d89bb4e6cbe54ef08537e9ad63baa0780ca6cf7ec53ee85dfdf7250f417a44573ab0606171f2c5"
        "a9028c70b5ac16303b30df645e99037f14244cae58d53cef34e112a5106a0de997ea8023d82cd2fad"
        "e2497159042b2c36b4ed4b8b665df1c074a0cc511a7f93d5f205941ad64e6a025225f28343e2eee74"
        "275346545e5c39e720f607545012058f0484d37f3dd3e0874e280b19131d15064cbf46dbbcd1269e6"
        "717515b62526d8fadea26bc9c780b37f35f409f4b13b55311d6ae410c9083e024e70c5348253dad80"
        "5a27584f4e1b319c2cba375f607cf7f34dbf18c10dd023e5f25c566373c0cc187d95ec374bdd12a70"
        "8f80a1bc6c413f72ed770f9baa48ca965a2de6a653a151773797759022b8221b649769f32cf0d69e1"
        "eee14c0e4dd62eeae701b81ca850413ae4696969845a06bde497b352e8684a933fac06c2057659140"
        "4ec9b5e6152fa10e55c5c1ec9f5c552837a1d44b60b95ea4e6ba98d2f43c4e2943d788b2d0c1ec311"
        "a9c3853530cd1c51348b3cf8f6b8746a34f4dd3be8d8bb60fd1287a9290bdc900f3690d2fc7a22920"
        "892b9dda51f5d809952bed94dc61fa36e39619f2a6d78c6df353291f7520b595be5aa8d85320aee0d"
        "214ea93e76a760a4e11f8797c12be5a9d1ff6d1e9826cde57ac8981c3ac6d9ec5efeec5618103a04d"
        "89245ec5d52f09a4d7069dabe629b810d2f2619d847c1bc1b7b164a3afc7daf1cee354af5aed20684"
        "1692f2de8c528dd2e609048089120a4f7cfa3c44718d99e64afa1a548b40d5a35c7922eec9c3bc96b"
        "a0f79cb8cde23787da3bae67ca3f5decdb1aa38229454ad829bb1d07354596da5fbbf1351771a92fc"
        "abea8628211218d6d8b92bc17066dbd4e2c7c355903bec92b171f273504be2a177e6b46b3622"
    ),
    "hmac_sha256": "3ab89125b1065690656849d0f80af9ef18b092665bf8c279015a79e3cb4f5a98",
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
