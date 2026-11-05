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
        "431886d5c1e521d164b7848a6365fa1cb5bd2bbe3e3de39718993782d237ee14d424952b0a9b98181a73"
        "5040c5e9e5af4ef443adfe167e007342dc57674afed6a9a7db6ceb3314bcff8463ea22cf8fa83b7e1765"
        "7012bb9a92f4c1ce3d856c89d48f961a5eaab1f80546ca9e239896a924f96278f787dab172f407721f93"
        "468216fa4bc83471bd590ba3aa072dd23171f0063cdd99ff5f9ed149e9d9c8bc7f327b20e5f30cfe137c"
        "e051c403dcdf6bd4e8d0b9ae454aaa11d0548d2d92aa27adc3a46954ef8880ec24fda52cf8f8f946e5ad"
        "68bcc11bd3f1e2e35da0f0f1167b1b6ed2da6315f83281a8f490236996af459e52deabad55d1eb9293a1"
        "809d61232c419c27f9eeaa4f08b2e8af4e150267b6452019e6688dff1b07faaf837745326c9c81252df1"
        "f743dd9e4d10186552e2728713ed464e0302992214779324b765ea253ea1e961db03f12ef11b61c0ef48"
        "678d1c9074c101590e6f6813ee8a055983790f39683681d6ba6fc3c672715fc4d5fe8ef28be0e698323f"
        "d4723555842beab8fb68126bfdd941ed53a3b155a0d0f4e0c1223324239e741fc59035ea93ffb4103c14"
        "817996c847a4493d553c1324b64b988db9449136e858132a7097ed4891ce3ade679672fbe3dcdd0debb8"
        "85a69653bb49a111ad71a4bc65877940a9c68ada6e58d1a37bc8495e5e866be67bdfb8b39c8a74585d0b"
        "d0ea10cd659921a8060f0a04013f6a4103506b694b1429920f035650db35175bcdb8cf9dc2f152758858"
        "a76ba6aa6f6d1874730be790e2fa1e28f9149fbfb9731ce5d97ea9df11b1262a8d79223930ed630496a0"
        "3a56e0e2ac1f6778da3747673b282416b331cc2b816600459f7e0a40c381823712af8f56dde5c8601a7c"
        "5d638e25f690dcc67d2012d3e6f4f855b7121d617ffe64f8b91c61b800d0d27711ff2a5623c7f57400fe"
        "02ee129046149a6b17e53f719f78471786d60d8db19b2e790858b20384e09ebb536e580051f2d66d9710"
        "39af39f8dcc8158e6114b06a249b36fd0b4ff4f85abd768485a93df2414bb3c28bec408fcd24441d8f67"
        "6ce23b8cafc4e3c4bd139c2ee22dbb143b54f0caf390765815a26f694bddc5b7c47556cb878ba3159972"
        "9430f6ac669e9dfe0e65dd6ebc8a851f53e9dc844b3fce8ecf61a4b478107bd5e8ba30430c48e57b8d91"
        "510c980d6e1cc506b31c2e9341f775549c0e87e112e0b8c84066977f669f48c37fa83e4c7e157f975cad"
        "548ab4e5f1c3ef89945fa11493f4ad7901206e9fe2262d11ee3a2f1324ffce1463f407d490ea2fb8f733"
        "9776a02594903cc904236a1af734fca5702833aae6b49c9748d4cbf4ca3344e9d4b44c4d2ce11048cb49"
        "5da0b7cb275d57cb90a64c6c54c87c349b1490891033ed13b154ad36ccc80af06551f02d5c10026ab735"
        "2dcf16af57edda3ed4cabd1d3d49b0c7035e293c16cae94a895eda1ef0727fcb6db5b93dc0aa33ea675c"
        "d6942dc6467096a3d2946058b62d6239ba2490dd0a874c3168eeb2e660614c58da936095a677844873e8"
        "db119250aaebeef2ccdf701db156f1c4c33f554d2b6dd74b5cd21e3cad3277dc5f1dc4b99fed1da273e4"
        "dd9c5f67cc761d7997911f056afa786394e2d32496a013979f9b0bb20f8363333cf97076ef8e34f37cf1"
        "3e42750faf99b5a1ea4e1e47e7368fa1482363573ab041a1486616e4588c53aa11691d844718c5e8ad00"
        "fcf163f5b7cb1d78b112ec47a6631a3df07aed78d2d4081d11e2747703038acf8f6faa9a8dc9df990089"
        "a2e8678506c7379c752a9dab33cd441083feba805ce80f43114229eac6c183d4476de4fe285360be1ec4"
        "545e71a439baf0a7cab7b92aea181d72736c9cd87e6cbf613b6c49373bc8a853465a040a7ba4acf7b01a"
        "bc5f0e9bb989563ab319514db4290fa00468d639eb64bb8d1591c616dc2e813a2fb3a31f72e1fb09c2e2"
        "238fd64e2b1c5238a6730f4990bd47b2c8165c7b46110b8abae90bf9989212a045f991163fabc6e036ee"
        "0a62d7d9e316774ed5a38a2e2a15fe6d9f22c141e674668318c911fb7edb8769430d772783b1a5214a66"
        "9f55528e537bb281bb23ed701cd8365940cbd2a8b9c464875018864ff6e8e1e8fe7db94bf5e2bad4ea7c"
        "e6cc254e0ce57093086f064b29c3d52fa7b70970c4bce8792cf0c66590d82b129b9edad20674ed48716e"
        "59daa322a92d80b45080f219f774c15d5c42c5ce308579bfa1b09199c188fd44e69e127c3540ca71a5ae"
        "c504984bf89cd4e1d2f10e06c12e791d3b3e01f292a066fc04a8728a95d79b14eb0f679087f1bb7fa1c9"
        "c3cda0fbab4161908288e670d18622cd4aeba7d16e24e9787b9aa952c058ba9f64c12ac6a83bfa9972ab"
        "5a5f81ea2df4223a780a801624eaac9fbcf1a9747429054b22b2f84e85672257dfd9d4cfaaaae830ce36"
        "9c9e8fa2c47288f764f25c87f5dd39aeb21e04a90bfdba8da1ba6c41927fc9f191803fb1f30ed9984d5f"
        "04bb0a3f5566d2fae887b07c851f5b97a3940f3895744d50a7b7f71bae73fbed5995f6d163258bf36a81"
        "763351d3eb475f58cb646dabbc28faae6115b8ec91ee537027803f5bfa13fc8810a7c424d60ef5e8b46e"
        "6cb52eb9388ba8ba9d0ad49e750c28ff6455635af91b8e897e926ddf028ad899f65beb563029db03a787"
        "e89b0a50ea60585707c2e4b0a7eaba4a9f8af141cb16f0c9a254f86860326b7be8be2a684eea9fbc0e6d"
        "7ea804e0a166938f860f1b4e6c7beddbc4ad1ed5bb81194059918a9041384c4f70dac701234e8cc5105a"
        "82174c59fe08072c65078c35d9f0eb5723f026f8948b56b1f832ec2bcc9d6b34a68dda8185fda14c440e"
        "0c04fa29b303b22f89038619fcfc1968178143ab2e647d711e570a3f17c6474d2e786765bd66e516947b"
        "28549d38625766a3e82b1433a4109a282661a4257aa7b4be9607a6c0d8a4c853cb41439ccc74fcc35353"
        "bf158e8c2f133a44ec5f8d1217a30a575c769af64f095eaadf29a27f2faaa26b93a5c667bc8e1bf3b8bc"
        "d8dda6df90c801bd7ff1419e4d46f59c97f8b0c47b5994474703072953980593cbd6b529cb83df1fd4a6"
        "e4f709563cb32fc1ab929c636e82ae6d9da38bdd7e3caeae25e8fce33864f34f5cb73964822ac223e5d2"
        "f523273b199e2a46274b4ee33808f5541d8d0200339b98902c250f70b891b13633c384210ea7c332e53d"
        "ad3b537186b822a819f06f22ec308fe6e798f90710255ad28c3813bef924d8826a80d67665c75cc6a98d"
        "d0a57356b1a7e14c82529e0e867ff6848b9cd57928b1dd45a621d106a04b64dfcd7ba46f385afd599646"
        "95250582fabb93ddc0e581c8c97047a46d7131aea8c7f4293bea3b6ce9b2648eb35565388de0b6f6e349"
        "8614cf9df05e9f9e30015982737e4d8350f87b57bd5e88e20e12b1715e5be23a4e1187798f88d80914b7"
        "1398a06f1cdaa693334a6e7865e378b37f1f818063c5f3ce41424f58429a5773e80eea955ee827a78eb0"
        "b511aa8274d55967fb4ca416b361feaef1b5a83730b74a5b4042a7d8f7b4c7446f986b60c36cd436ad73"
        "3c0ff03d619f306f8cc4984987a67f843ffb82a4047e2a83490345847895d8aa34d3e8370d033445458b"
        "cc51d84cb055c61a9e0a17b8014488602932759ff2c54418ff7e0aab9f86d542487e0c9ed3b913886c30"
        "c0def0888eaf52051055febe1776bba786eed63ad34769c93b88687cfda0127d446d5da9d6d3c70e99a2"
        "ead5acac1d8f89bc717d43187e93246f04fbf6b127a5467e330b17c18076b3af3426c370b4b264027dae"
        "71a84d9457012fae4664ffecbbdca59712067c7e2a049584ea7fb06569af937365ae02ee85e4502b74dc"
        "d122149e024972c41762032364d9dc7c1edf18db6a64494b7b34388f37638eecc96cd358960f9915fe80"
        "02bc27a895dd146d18efca7858c937e5a871e978b5c17efb5b9a7ea822716ade50a229516a4e2cb6cb45"
        "14dd37c7aec3e46f2e681b2156ae373df395257a77c9e268685927836c56a9ec272a5b761c81d8ef02ef"
        "2abf248a7423b35ff3f98f2c2b184ff54bb79bb18cd386d5ad2e65830c69d8ce03ec160f30a62d9fe0ac"
        "2ee4d69e087d2b2eecd409bdf6861d1d51ca1d82ebcd4ff9ef0ac308d7d5883f6d310f8184c4f8f7980d"
        "449167dbb4396d8042b8497e25b8caa12f69759559a88a28f4eded4378f4815e7a3e95b60ba2e7767872"
        "16a6c92ff758347f61b405a04ba6971745536af2f05b68f2dd6edc09d737b244fb58b874e79351488363"
        "4245cd1a92ca11ccc49e391e7327bf18ef7c666325855d504a3ec6e8c6244e0b681f311e2fdd96dbbaad"
        "f7148659b6b8390c99e50f1549b8d1a6abeb9a81f732c4c81d613547e2818e70e2070ede69c594e4b2a9"
        "b0b53e1fe2840dd77c3cd94728a1bc6a0969fb08799c761aeac4dc517ce4d9288521068e000e65924700"
        "8785d965d1c8681d4458dc41ddfaf82a2bd80d00216bd5b9c9e47a23ad8fb02740642864bfb315064461"
        "001e6f9168ebb0df9a1212b25bb5393f64aee5e917deead712dddee86a18ef5d43de0159f8663acdbdb7"
        "73b2b174aeaa9544f6578c43b40d23501c957f84c7bfb6e9dbd5dbfdf9c3079dcd1704a6c447f9183149"
        "13fd706ab493e8bb707fb47e395443334116d58be2470b5f900d9363b91e9b5e08f34cfe4ee69309268a"
        "7ffd44cb1800bb5b9a701d55ed5c8c6e745fe64414b84d1ca0e93c7f4a9faf072591333c6c385e5a1382"
        "cf29bfe325f71e89cfdc99598af45cfa2332ead3bf8621565d9edbb03c03fd348cb1bd1782b80bb173ad"
        "1e9999672015d08ab0307ff631d2882285ef51e983b8250ac11f5bf26812f237fba8145b4f5d0ed8f987"
        "ff322552482c25f49ba8eca7785cf23333966f5a8e7037515b5ccd90190a432149a35a1ce1fe4bec654c"
        "1df7651a0e936e9a34e6cea580619213a30f0b812b64d2c5da7f37857248d253e64096b02b3e293042ce"
        "da2f6a2b00620c84e3a297086073834739e89dec29907865bb38a84b9bf1f58d804f6d5efc462fcf780e"
        "656c1ee28b147a1ef76125ff5d3621ae2f39bdced09c1e0683492e239887e395529c07bb60d060dc0c27"
        "a959607cf7c72cb64a47540f247a0dd1351a4f81530fbcb72a5bfc02a92bdce4488da8a0ba67f5b760e2"
        "12b2c810a0a081509ed28c0ccddfa2859badfad7922f77856a54936ed5f0f917400b709feabadab66bca"
        "552070609bc7c8f5732acc90986998c00cb8cb90b8bb628faab44eeec59370a991255cf2bafc3dcbe75d"
        "1825372deaa8c928be0d9140a3824ccb7b2f5299ecdcf3b3977723ecdb61bd44fad229b57f0d2d1226de"
        "9bfec8eb542466b85e1ca0280b74f3ddfb6724f4b090c1a88423e4a2db6140f49837ddb4dc04cda9f7a0"
        "a24bdb78194ee2d8d4a11b545c8f4858117cefc75b4d02223e100ef69620afe348f434fa4a5f58701ffe"
        "d14d3c68e5a9599755097f0adcaf82687b28c019ebba1393c3a47f407aad022aaf97a7e01d6933ea5c58"
        "a1187fca385b2211b8dc4dbdf84cf31c412540ca60382b45ae9ad48ecfe18a379fdc4da9bd1282096569"
        "4d416a86f40038c5e9b11aaf71"
    ),
    "hmac_sha256": "578e9669570b936399d823ef2901d0e55eba0361ae7edfd7482bbe947d2e4315",
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
