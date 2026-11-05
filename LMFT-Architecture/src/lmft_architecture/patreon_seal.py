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
        "431886d5c1e521d164b7848a63656e798bc13623cf19b14dd8646992ff9fff44b58ca2591e2deb36b447b9"
        "62f65aae9f4ae9e8a39ab1428e6e9bda35d35dc9607e886f29a4de016fea84373b70dbdb1d9e89921ad8ac"
        "0df874d9ce2a7c956c0a94af961a5e8e78e3433124f33c0a297a613111778e19cafbb92ab952997f31e977"
        "df5075a667b1490ba3ca37ddd35171f0671cdd99ffddbed1c9bbdd90440791803ad03e28f21299c4f626c1"
        "d0427d46ca84cb7c373b3b92d14b8decf2aa668dc3a46a54ec8888e022f2a52cf8f87947fda9489c4118f3"
        "f1e2e325a0f4f316790b6ed2daa315f4f2a1a7f69423499e2f759190deabad57d0631192a1809d61272cb1"
        "bc37f9ce66cc08bee8af4f158aa4b755c058e6688dff5b07faaf8d773c326c6da12b2df187431d9f6d1018"
        "655266718713ed454e0742192594779304bf65ea253ea1e96edb73f12ad11b61c0e848478d02907441818d"
        "a153a4c736c93d39db7811546614261b0f475cb19ff1385b9aa4af70e9a02474fe9633d4a5bcaa19becc69"
        "fc3171bc9e19e7501ecdaac56055d3ae7ff34eeaa2cc2884b73c06398f6f635758ea45eab32953cc07cc37"
        "e3470c1d526a6480fa4d60c2647c800daf21765bd9b67cf458f2e5b2ff94beb3b6f2b78e71fe3d23abc6d1"
        "0f5b39eb27c091b695641d9b8558feacf2ba07289d57623429350759a1ac6461aa2ddd9c23f745fec7ef9c"
        "683bd335df6526185a2dcd808c2ae034fc4db00efbcadf3fe3486129a53989609e234fe275e70604ae8a44"
        "05a1f5925d8e083141062299d7e14ddaa033e7808b73ba080cf5181e223096b9bb04fff648ad5767ee634c"
        "e1d8ef4bf27c624defa034579006261bc2ac582e3975121fc5520103a12fd25830feff0312c380e205dd7b"
        "77c8dfacbb59e81643273564e3c379e7b05300822eb32221c59a18d15246aec2f4cc6dffe34099ab742148"
        "4eb4f13585417e2ddd8fa5d9f2dedb2c39aa8a65aa8cf2b27ba5aa7f1b44ef16275778f57259ffc7db0d92"
        "48eb9702aff8ad8de94be8294fe79131e90acf0f729857744f1e15686cdf15b2162b6d40ee2b2e7a1eca62"
        "d9634ec5035ad65b48dfcbeadd13f763300183c3bd028317757ab30223d07c90a42433888e00d5b6ace54d"
        "caea4def746f06ce9e90b64175e51ac6507e9a20372616721355155fb947b0342625f5d96ac00893114778"
        "84bfe90e92eb2ef7593686fdd30fe80a8f6d8e7370394d1666731740bd5d7c68590bdb049d3840feec1b6b"
        "2069efe47d4709caf3e68d9265bbb2559f2c25143f3dedf5922d90e71cff2cc7e8610b01e08d46a9176cfe"
        "79a7ec3f1c8b5a8a96c001cd5ef3418c8e659264e23a702e3438a29d6ed6b98501c414568135e3872cd736"
        "723137a7b1e88cf6170e4710f01ecbad8c673fda73619b9dcfade8004b53cd1bec9d29d0ffbacdc34d575b"
        "dacb7a453390c992ae00bda188040b79afa8f1be6ebfea3da0ec55d34f45d87dd45d85902425208c8373ca"
        "3518768980e4f90a9464a844cf9e4d6aef78f6b46be46a5efbbc18cd6b6b58cf9fc7355a626557d79a5a8b"
        "c13ffa1d2cedfa41c3bd12c419f955877218560afdebb7ce8a045323952d7f940ae491ad72f3195d06241d"
        "07edb4678b865187aa444a987e2446b4ec2e66f373415c8bccbedeb38a2f042c42c7b8509c46ce3af310b7"
        "8f4ab7820bb91ab4737ce7883fa7a105027cadd7e81d1f4f5f4b39a980f0d144f2b4f286dd653ccb411b94"
        "065bd0971293e9a12b9faec82d3bf76024937c8d20ebb05e392d0f0d7ac73938eb43cf48fa9bf53ec06f5e"
        "d5231bab099361c415b38f2f195f24898eac7f4fb905d6f4f8d9c737e87f1e23cf449aa788c3b2b6435666"
        "4c289d95f81102a01cf78e5e9040995242645e79d9851f5e41dddf7710b86ec7302591da03259ab268468b"
        "5ee9673414f50aac2257ea15184310c3b947df86d5f60eded370867613c1f3275250e09c40f3828239dd65"
        "532eb0e0a98be293d3ab83a5a1e1b0c845cbf3d0f849c6ffb5f04e009983de215bed784eb8c8283b30a01e"
        "dedfc96ae46d226c8e6920a4bb4c802f92ee6d92bb5c2b1be093c3af6b1963ddedd18041281f1db5d9ae08"
        "5ac8d6806f4c50b8abad333bf8af9626262b873d09f0a2701b9de2e2ca23adb6b13ad6c0ce1eac60eeaf13"
        "b364bd69440d88901b52a5261395ede0c19f4720f49427ff2c9b36d9765702a3ce64b7b458eacc75adc0b6"
        "81e345c19a51866fb198aad746012a965920b56088d8a2620ebc66f6ae7e7a64f3466acb321c52d46e7aba"
        "adb28c84600c50beea76ffbeddf7028a65acac2251e93d6781ea8494944d9f382b1e7bbd26d98f36757b0e"
        "836fb5832dfee5f6d585bf3fb6f4107cac48ba239aa4b2122138688e2fbd94fc991be33924e488c7e4a7a3"
        "62100f1a83a533a040ccdac2dc0d6bc6a3e4c72cd11204e81cce27d4b03c2e2c25da6f74f4851cacd290d2"
        "17d7c119a56829e06e82c16a16c7587f3256536978ad57b3b7f56639b835a1d0312d9a2011ced9c5ddc14d"
        "6a522d1f32e6f4c7d7a5e2d7f614679e47da1ca139be5282f3a283103627ceb864f10af039ee9985257114"
        "42a3944a77bc8252d9e86e23afaf4467886d8fdf90d22dc29f0dc7ed1ceccab1590cad0b471d28089f8e62"
        "cb645eb36d7b1f43f202e035bd264472b4ae01f6ac697bc83a22201a7a75a8e0ec070bc7c1123de9baa2da"
        "c21d023a418f7dd3c2a3670089691b045ec075ded5dd1fb5dd1d34485da6ec0a67c865eb8427daddadff5b"
        "096021e95f75378e2e801f37ae93f5d5ac516d4f8dbdb24a20e963883496b7e1b7d769c44e9d8c2bf102ac"
        "e50913783d9730c2dde4471a974240f2ec38ebe54c9106e8887075b3d7f1288e4031c5c31ab61f18d31c0c"
        "6a015a31f3ef33a67d07beea80d67d601848684c7545bad196f51e96fe646bd42a794be4d1ab3532a29acc"
        "fac6c4a76f3f0c72df3ca18e908f7d4fb7e9c08c4a2ea09bf294f269bd50c90c53c108b749cced49139fdc"
        "0af532198d2ddfa456f85de55731fc8661c2e64957b38a72fbc1e42241ef97f35ff6bae2c92a6311890256"
        "bc889a75e62c60c6ba0f98fff68cc101859c279c7fec0b084d3e843d2beff26840613f17412817120e917e"
        "31685f185d799fc485fb60e44d947b1da721f5bcec8e192d1139ea2d9babb4658c6241bf323edb212a5eff"
        "2258267d86fcfd7786321397019500b6ab540bf5138a2bafad95ea15d0a661af421edcce95ebe7f56c80dd"
        "173822fda3a124424b404344a4515b6bacd5c5b8af0b38a4b37ff2318697ab91384c91e0e92488a4d5df98"
        "a8a9ad18a8ea17d87a89dc29782d3c2bce2e78fb2087a98bfbaaecbe03013b6be5a060645f43cfb65d6e00"
        "519f3e6d2d90d207e5ded46a74be8044915692a34556cb7d975618cd4a97eb32f80dd510765ab8cdbf84c6"
        "410d8601c38426caa1a82f1bb51ad3e5df0a963cd2368b4fb0b234584f822b093911247be63fcf96a9a5df"
        "b091f92eb5b93955406ca4c991b92be34a3aeefed7f182f3004327992c7f8642f13283b219faeba0533179"
        "55e160e4883d1895968c21c2fb7cea44dbcf9c0c41ef0d76c26821c6049422bcec4258e65be8583af51c08"
        "a7b76dc3ba66ce1ee4e49981df26310a45d882376c13de0aff129607b5f24cfe7879b5309acf48584b2db6"
        "7f358936708697deadbe2ae222f5f1d3d0b18cad1a90188395e8a42b6f14053edfedbbd519bec348b15c3e"
        "f796481328d4456f99085330a1fbbf6742dd9575e8c9d2db50cc410b869ab60a910ed63ea85537d36692b8"
        "87beaf545cebd49fe64f7450eebd98bdd465a454a6dc442a70c5840c87be6962c2ec8e1d6aa7f159061075"
        "ad3bc7e03750590ea1856fbff639f75518ef71686f0f407c0ec9f3291e88f978e3f2e98491a4490751765e"
        "a52a06c259a2c4fcd4acc5700831d42b304d0d7fa0dc36f3f09ddc4edf93afd7f98c046e5962461718fdbc"
        "0a65d820627ea9a17391a250fe57c684e53f3b845f4eee26ebdf995ac87f80f3d43b01ff2588d7fff035e4"
        "3ed9169b46623c43134207e46ec2e257a3dbfa8068d218e61e3dc1ebb49ca24d30e3898bb24673bddafde9"
        "651ee22ca9e184fadc38590491190965d9ee4208a670433d59bafbc3242488e3e7d3423cfa59a988611f6f"
        "72a3e4bde0003c87a15ca98630dd60f267f13b5e1c85db483cd88476f072438ea6580fdd4d56785bab0806"
        "0e6cbdeab40ad71ac71e5ac3a8167e510890bf840969813cf096d7b4c36c31b2bd1a5f74654a4ca977450f"
        "97ce478ba56573d345420a1cf359614bfba64ba1670856e4c913266453b52019d2d4d636e4706ce8515156"
        "1f29a9603d01b77fdb7292fb7bfcf6014aa24ce6838a175ae116542071d42f2019b7ad98707a20b19f1200"
        "64aeaa6ba2a020b3e07aa97ce2cf0d6af6c4a62083e8f7754eb39f49b9c3dba7217ee08ba86450c3b3edeb"
        "826ee14051cd7c67588464435eebe441830dc41d45f67db7ea6cb7a6053deb495f588a7a54d80d0b99cc4c"
        "5eca0694c4695c2676ed84c410ff4318bbce57e68db87ab111b3922038a9624496dff44fcfc2beb12f4ce9"
        "2717af3720a63fc7e0f0dce42fc3f48945ab8ec79064d36e8fcc0690d2e8fcb66701692c70c7160db58eb8"
        "603dd995aa161978efe20de817dcdf1cb5a564086987ca62a73a318aed6cc8317f7199198b2eef9d4acc0d"
        "4c3d43203b1dc74e47e571bc863860d4b976b75be301a461a357e2f9078f00b521453a3b3934ee8b661a34"
        "b3e3855ef7cfce95c13380d6734842c20eabb2fa199232a491fe2dfa0bb739cdd5bb2cecd83630f3d92f68"
        "94752332343b0d7d766d5a4071eed53d8dd3c09bf05de520a63e65a447946e82750c5942066e92ec736380"
        "b804293ad7365fcbe1df4fcb0246999e44d526c9c3fe7e0675796f81e996f362b0f2ac3c8fb694df0a5c35"
        "22a59197f2f7ad0da1d4fb3beb272df68ac1796c2f7c016459a47f38d8f37d615f0e416ffedaeca20946e7"
        "5927f0a40fc95bec07bf00b98c1d7df22a8f2e3be46c1533f2599411668b1092a3f755ff481c9d88955635"
        "2a3a5e9b78417039f62e730534e6036aae8019661e8dedc790a55fc55f856e6e0d1c0208c6139fd231dc3d"
        "1d07f0a8513e3c490a295d3397ff5d44f7abb49724f4f9ea820f847238d900bec7472f7a5102c097f46fd7"
        "4f5ac2923e8023042cd939de21d03e1c4790c915c271e4b8b81652bb67e4ca3492df38ec92af1a16775821"
        "4f3d0f6d8e738a886b1c2ec7d4c4633b236928a3ebc95fac6612d56f4c4df54f5cb6bbe1447b181a8eaba1"
        "087e2760a8acbf08c0ad1fbb0442058c78c4e59e1620c068ce0388bf855341dbc422a1e439d20d043169a9"
        "78cf93b964ca44696a260eb23fa31ec57dc5c11aae022d1b33fbb42b19ce6809779355c664bfa482675a8f"
        "0b78ce91d21871e2ef6a785fa23e72168aa138124c2e8c65c25669e560f03b2d1d2db96cb96e3f723c5a7b"
        "1f58ed23a267e60099b1f07a4660e4ea895e1c6e10535ba6a3a2eb91df2f8a18a84fbe43faae07aaa5076e"
        "1d5fdbcc36ec08373272ea0d7bf032be94c65002c9fe0c47f6a2ad8130ee34adc030adbd56297bc6767b3d"
        "559b5f"
    ),
    "hmac_sha256": "89ece73fb77c8314bbfdb75b95c713914eb15fddfc2fd5aaefb5676886f5707e",
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
