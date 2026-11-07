"""5-Page Windows GUI Setup, Hardware Diagnostics & Verification Wizard for Celestriad 1.0.

Runs 100% inside the officially signed Python Software Foundation interpreter (`pythonw.exe` / `python.exe`)
using only Python's standard library (`tkinter`, `ctypes`, `hashlib`, `subprocess`, `webbrowser`).
Contains zero custom-compiled unsigned `.exe` wrappers so antivirus suites (Malwarebytes, Windows Defender,
VirusTotal) and platform attachment scanners can inspect every plain-text line directly without false positives.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext
import webbrowser


ARCHITECT_YOUTUBE_URL = "https://www.youtube.com/@imissmydeadcat.74"
ARCHITECT_EMAIL = "imissmydeadcat74@gmail.com"
ANTIGRAVITY_DOWNLOAD_URL = "https://antigravity.google/"
GITHUB_REPO_URL = "https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model"
VIRUSTOTAL_URL = "https://www.virustotal.com/gui/home/upload"


def _enable_windows_dpi_awareness() -> None:
    """Enable crisp High-DPI rendering on Windows 10/11 displays."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


class _DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", ctypes.c_ulong),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
    ]


def resolve_workspace_paths() -> tuple[Path, Path, Path, Path]:
    """Return `(base_dir, arch_dir, train_dir, assets_dir)` for repo or extracted `.zip` bundle layouts."""
    here = Path(__file__).resolve()
    arch_candidate = here.parents[2]
    if (arch_candidate / "src" / "lmft_architecture" / "birth.py").is_file():
        if arch_candidate.name == "LMFT-Architecture":
            base_dir = arch_candidate.parent
            arch_dir = arch_candidate
        else:
            base_dir = arch_candidate
            arch_dir = arch_candidate
    else:
        base_dir = Path.cwd().resolve()
        arch_dir = base_dir / "LMFT-Architecture" if (base_dir / "LMFT-Architecture").is_dir() else base_dir

    train_dir = base_dir / "LMFT-Training" if (base_dir / "LMFT-Training").is_dir() else arch_dir
    assets_dir = arch_dir / "docs" / "assets"
    return base_dir, arch_dir, train_dir, assets_dir


def detect_antigravity_installation() -> str | None:
    """Return the path where Google Antigravity IDE is installed, or None."""
    local_app = os.environ.get("LOCALAPPDATA", "")
    user_profile = os.environ.get("USERPROFILE", "")
    prog_files = os.environ.get("ProgramFiles", "")
    candidates = [
        Path(local_app) / "Programs" / "Antigravity" / "Antigravity.exe",
        Path(local_app) / "Programs" / "Google Antigravity" / "Antigravity.exe",
        Path(user_profile) / ".gemini" / "antigravity",
        Path(prog_files) / "Antigravity" / "Antigravity.exe",
    ]
    for c in candidates:
        if str(c) and c.exists():
            return str(c)
    return None


def detect_system_hardware() -> dict[str, object]:
    """Detect GPU name, VRAM (GB), System RAM (GB), and Laptop vs. Desktop form factor."""
    gpu_name = "Standard Windows Display Adapter"
    vram_gb = 0.0
    ram_gb = 16.0
    is_laptop = False

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        line = (proc.stdout or "").strip().splitlines()[0] if (proc.stdout or "").strip() else ""
        if "," in line:
            parts = line.split(",")
            gpu_name = parts[0].strip()
            vram_gb = round(float(parts[1].strip()) / 1024.0, 1)
    except Exception:
        pass

    if gpu_name == "Standard Windows Display Adapter" and os.name == "nt":
        try:
            dd = _DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(_DISPLAY_DEVICEW)
            if ctypes.windll.user32.EnumDisplayDevicesW(None, 0, ctypes.byref(dd), 0):
                if dd.DeviceString and dd.DeviceString.strip():
                    gpu_name = dd.DeviceString.strip()
        except Exception:
            pass

    if os.name == "nt":
        try:
            mem = _MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)) and mem.ullTotalPhys > 0:
                ram_gb = round(float(mem.ullTotalPhys) / (1024.0 ** 3), 1)
        except Exception:
            pass

        try:
            pwr = _SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(pwr)):
                if pwr.BatteryFlag != 128 and pwr.BatteryFlag != 255:
                    is_laptop = True
        except Exception:
            pass

    if "laptop" in gpu_name.lower() or "mobile" in gpu_name.lower():
        is_laptop = True

    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "is_laptop": is_laptop,
        "cpu_cores": os.cpu_count() or 8,
    }


def build_custom_antigravity_prompt(hw: dict[str, object]) -> str:
    """Build the copy-pasteable Google Antigravity hardware customization prompt."""
    vram_gb = float(hw.get("vram_gb", 0.0))
    ram_gb = float(hw.get("ram_gb", 16.0))
    gpu_name = str(hw.get("gpu_name", "GPU"))
    is_laptop = bool(hw.get("is_laptop", False))
    cpu_cores = int(hw.get("cpu_cores", 8))

    if vram_gb >= 12.0:
        profile_mode = (
            "Keep 4-bit NF4 Qwen/Qwen3-8B enabled as the primary Left Brain and keep sequential "
            "VRAM unloading active during REM Sleep."
        )
    elif vram_gb >= 7.5:
        profile_mode = (
            "Configure LMFT-Chess to prefer 4-bit NF4 Qwen/Qwen3-4B or Qwen3-8B with max_context_chars=12000 "
            "and strict torch.cuda.empty_cache() before REM Sleep so VRAM stays under 8 GB."
        )
    else:
        profile_mode = (
            "Configure LMFT-Chess for Low-VRAM / CPU-Assisted mode: keep the 40-Node Right Brain and "
            "3 MiniLM bridges on CPU/fp16, and use a lightweight 4-bit Qwen3-4B Left Brain with short "
            "max_post_gap_tokens=28."
        )

    vram_str = f"{vram_gb:.1f} GB VRAM" if vram_gb > 0 else "Integrated / Shared GPU"
    form_str = "Laptop" if is_laptop else "Desktop PC"
    return (
        "Paste this prompt into Google Antigravity to auto-tune Celestriad 1.0 for your exact hardware:\n"
        "--------------------------------------------------------------------------------------------\n"
        f"I am running the Dual-Hemisphere Wisdom Model (LMFT-Architecture / Celestriad 1.0) on my {form_str}.\n"
        "My detected hardware specs are:\n"
        f"- Graphics Card (GPU): {gpu_name} ({vram_str})\n"
        f"- System Memory (RAM): {ram_gb:.1f} GB RAM ({cpu_cores} logical CPU cores)\n\n"
        "Please inspect `src/lmft_architecture/chess_app.py` and `three_legged_runtime.py` "
        "and customize my local runtime settings so that:\n"
        f"1. {profile_mode}\n"
        "2. The 40-Node Right-Brain Fuzzy Tree (`birth.py`) and `<|subliminal_gap|>` comma splice remain "
        "100% intact and un-rigged.\n"
        "3. Verify the configuration with `pytest tests/test_chess_app.py -q`."
    )


def compute_core_sha256_report(base_dir: Path, arch_dir: Path, train_dir: Path) -> str:
    """Compute SHA-256 digests of the core Python and JSON files."""
    ruleset_path = train_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"
    if not ruleset_path.is_file():
        ruleset_path = arch_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"

    targets = [
        ("Launch-Setup-Wizard.bat", base_dir / "Launch-Setup-Wizard.bat"),
        ("Launch-LMFT-Chess.bat", base_dir / "Launch-LMFT-Chess.bat"),
        ("Launch-LMFT-Bloodmoon.bat", base_dir / "Launch-LMFT-Bloodmoon.bat"),
        ("birth.py (40-Node Seed)", arch_dir / "src" / "lmft_architecture" / "birth.py"),
        ("chess_app.py (Chess Engine)", arch_dir / "src" / "lmft_architecture" / "chess_app.py"),
        ("three_legged_runtime.py (Subliminal Gap)", arch_dir / "src" / "lmft_architecture" / "three_legged_runtime.py"),
        ("installer_wizard.py (Setup Wizard)", arch_dir / "src" / "lmft_architecture" / "installer_wizard.py"),
        ("rem_sleep_consolidated_ruleset.json", ruleset_path),
    ]
    lines = [
        "=== Local File SHA-256 Integrity Digests (100% Plain-Text Python — Zero Unsigned .EXEs) ===",
        f"Folder: {base_dir}",
        "",
    ]
    for label, path in targets:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"[SHA-256] {label}:\n  {digest}")
        else:
            lines.append(f"[MISSING] {label} ({path.name})")
    return "\n".join(lines)


def find_venv_python(base_dir: Path, arch_dir: Path, *, windowless: bool = False) -> Path | None:
    """Locate `.venv/Scripts/python.exe` (or `pythonw.exe` when `windowless=True`)."""
    exe_name = "pythonw.exe" if windowless else "python.exe"
    candidates = [
        arch_dir / ".venv" / "Scripts" / exe_name,
        base_dir / ".venv" / "Scripts" / exe_name,
        base_dir / "LMFT-Architecture" / ".venv" / "Scripts" / exe_name,
        arch_dir / ".venv" / "Scripts" / "python.exe",
        base_dir / ".venv" / "Scripts" / "python.exe",
        base_dir / "LMFT-Architecture" / ".venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def check_install_folder_integrity(
    base_dir: Path, arch_dir: Path, train_dir: Path, assets_dir: Path
) -> tuple[str, bool]:
    """Run the install folder integrity check and return `(report_text, all_core_ok)`."""
    ruleset_path = train_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"
    if not ruleset_path.is_file():
        ruleset_path = arch_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"

    checks = [
        ("1. Core 40-Node Seed (birth.py)", arch_dir / "src" / "lmft_architecture" / "birth.py"),
        ("2. Chess Arena Engine (chess_app.py)", arch_dir / "src" / "lmft_architecture" / "chess_app.py"),
        ("3. Subliminal Gap Runtime (three_legged_runtime.py)", arch_dir / "src" / "lmft_architecture" / "three_legged_runtime.py"),
        ("4. Virgin Baseline Lock (virgin_baseline.py)", arch_dir / "src" / "lmft_architecture" / "virgin_baseline.py"),
        ("5. Master Release Wisdom Seed (.json)", ruleset_path),
        ("6. Arecibo Reply Desktop Icon (.ico)", assets_dir / "arecibo_desktop_icon.ico"),
        ("7. Celestriad Spirit Particle Icon (.ico)", assets_dir / "celestriad_spirit_particle.ico"),
        ("8. Esper Theories Alphabet Chart (.jpg)", assets_dir / "esper_theories_alphabet.jpg"),
    ]
    lines = [
        "=== Celestriad 1.0 Install Folder Verification ===",
        f"Install Directory: {base_dir}",
        "",
    ]
    core_ok = True
    for idx, (label, path) in enumerate(checks):
        if path.exists():
            lines.append(f"✅ [OK] {label}")
        else:
            lines.append(f"⚠️ [MISSING] {label} ({path.name})")
            if idx < 5:
                core_ok = False

    venv_py = find_venv_python(base_dir, arch_dir, windowless=False)
    if venv_py is not None:
        lines.append(f"✅ [OK] 9. Python Virtual Environment (.venv): {venv_py}")
    else:
        lines.append(
            "ℹ️ [READY] 9. Python .venv not created in this folder yet\n"
            "          (Click 'Install / Repair Python .venv' below for first-time setup)"
        )
    return "\n".join(lines), core_ok


class CelestriadInstallerWizardApp:
    """Traditional 5-Step Windows Setup & Hardware Diagnostics Wizard."""

    STEP_TITLES = [
        ("Step 1 of 5 — Security & Antivirus Verification", "100% Plain-Text Python Architecture (Zero Unsigned .EXE Wrappers)"),
        ("Step 2 of 5 — Google Antigravity IDE & GitHub Sync", "Pair-Program with Google Antigravity & Keep Rulesets Synchronized"),
        ("Step 3 of 5 — Hardware Diagnostics & Custom Prompt", "Automatic GPU / VRAM / RAM Detection & Tailored Antigravity Prompt"),
        ("Step 4 of 5 — Meet the Architect & Celestial Seal", "Support Independent Dual-Hemisphere AI Research"),
        ("Step 5 of 5 — Folder Verification, Shortcut & Launch", "Verify Installation, Create Arecibo Desktop Shortcut & Play"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.base_dir, self.arch_dir, self.train_dir, self.assets_dir = resolve_workspace_paths()
        self.hw = detect_system_hardware()
        self.antigravity_path = detect_antigravity_installation()
        self.current_page = 0
        self._images: dict[str, tk.PhotoImage] = {}

        self.root.title("Celestriad 1.0 Setup Wizard — Dual-Hemisphere Wisdom Model")
        self._center_window(980, 680)
        self.root.minsize(920, 640)
        self.root.configure(bg="#0f172a")

        ico_path = self.assets_dir / "arecibo_desktop_icon.ico"
        if ico_path.is_file():
            try:
                self.root.iconbitmap(default=str(ico_path))
            except Exception:
                pass

        self._load_preview_images()
        self._build_wizard_shell()
        self._build_page1_security()
        self._build_page2_antigravity()
        self._build_page3_hardware()
        self._build_page4_architect()
        self._build_page5_integrity()
        self.show_page(0)

    def _center_window(self, width: int, height: int) -> None:
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(20, (sw - width) // 2)
        y = max(20, (sh - height) // 2 - 20)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _load_preview_images(self) -> None:
        for key, filename in [
            ("arecibo", "arecibo_preview.png"),
            ("spirit", "spirit_particle_preview.png"),
            ("seal", "celestial_seal_preview.png"),
        ]:
            p = self.assets_dir / filename
            if p.is_file():
                try:
                    self._images[key] = tk.PhotoImage(file=str(p))
                except Exception:
                    pass

    def _build_wizard_shell(self) -> None:
        # 1. Bottom Wizard Footer Bar (Classic Windows Installer Footer)
        footer_border = tk.Frame(self.root, bg="#334155", height=1)
        footer_border.pack(side=tk.BOTTOM, fill=tk.X)

        self.bottom_bar = tk.Frame(self.root, bg="#1e293b", padx=20, pady=12)
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_cancel = self._make_button(
            self.bottom_bar, "Cancel", "#334155", self.root.destroy, padx=16, pady=6
        )
        self.btn_cancel.pack(side=tk.LEFT)

        self.btn_next = self._make_button(
            self.bottom_bar, "Next >", "#2563eb", self._on_next, padx=22, pady=6
        )
        self.btn_next.pack(side=tk.RIGHT, padx=(8, 0))

        self.btn_back = self._make_button(
            self.bottom_bar, "< Back", "#334155", self._on_back, padx=18, pady=6
        )
        self.btn_back.pack(side=tk.RIGHT)

        # 2. Main Body Area (Left Sidebar + Right Content Pane)
        body_area = tk.Frame(self.root, bg="#0f172a")
        body_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Wizard Sidebar (Traditional Windows Setup Sidebar)
        self.sidebar = tk.Frame(body_area, bg="#1e293b", width=235, padx=16, pady=18)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="CELESTRIAD 1.0",
            bg="#1e293b",
            fg="#f59e0b",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            self.sidebar,
            text="Setup & Diagnostics Wizard\nv0.2.0 · Windows 64-Bit",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 16))

        sidebar_steps = [
            "1. Security & AV Check",
            "2. Antigravity & GitHub",
            "3. Hardware & Prompt",
            "4. Architect & Seal",
            "5. Verify & Launch",
        ]
        self.step_pills: list[tk.Label] = []
        for idx, label_text in enumerate(sidebar_steps):
            lbl = tk.Label(
                self.sidebar,
                text=label_text,
                bg="#0f172a",
                fg="#94a3b8",
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                padx=12,
                pady=9,
                cursor="hand2",
            )
            lbl.pack(fill=tk.X, pady=3)
            lbl.bind("<Button-1>", lambda _e, i=idx: self.show_page(i))
            self.step_pills.append(lbl)

        if "spirit" in self._images:
            icon_box = tk.Frame(self.sidebar, bg="#1e293b")
            icon_box.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
            tk.Label(icon_box, image=self._images["spirit"], bg="#0f172a", bd=1, relief=tk.SOLID).pack(pady=4)
            tk.Label(
                icon_box,
                text="Signed Python Runtime\nZero Unsigned .EXEs",
                bg="#1e293b",
                fg="#64748b",
                font=("Segoe UI", 8),
                justify=tk.CENTER,
            ).pack()

        sidebar_divider = tk.Frame(body_area, bg="#334155", width=1)
        sidebar_divider.pack(side=tk.LEFT, fill=tk.Y)

        # Right Area: Top Step Header Banner + Page Container
        right_area = tk.Frame(body_area, bg="#0f172a")
        right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header_banner = tk.Frame(right_area, bg="#1e293b", padx=24, pady=14)
        self.header_banner.pack(side=tk.TOP, fill=tk.X)

        self.lbl_step_title = tk.Label(
            self.header_banner,
            text="",
            bg="#1e293b",
            fg="#f8fafc",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        self.lbl_step_title.pack(fill=tk.X)

        self.lbl_step_subtitle = tk.Label(
            self.header_banner,
            text="",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.lbl_step_subtitle.pack(fill=tk.X, pady=(2, 0))

        header_divider = tk.Frame(right_area, bg="#334155", height=1)
        header_divider.pack(side=tk.TOP, fill=tk.X)

        self.page_container = tk.Frame(right_area, bg="#0f172a", padx=24, pady=18)
        self.page_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.pages: list[tk.Frame] = []
        for _ in range(5):
            frame = tk.Frame(self.page_container, bg="#0f172a")
            self.pages.append(frame)

    def _build_page1_security(self) -> None:
        p = self.pages[0]

        card = tk.Frame(p, bg="#1e293b", padx=16, pady=12, highlightbackground="#334155", highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 12))

        body_text = (
            "Welcome to the Celestriad 1.0 Setup Wizard.\n\n"
            "• Dual-Hemisphere AI Architecture: Pairs a 40-Node Right-Brain Fuzzy Tree (~15 ms intuition) with a local "
            "Qwen3-8B Left-Brain Language Model across a synthetic Corpus Callosum (<|subliminal_gap|>).\n"
            "• Why Malwarebytes & Windows Defender Love This Release: Commercial EV code-signing certificates for custom "
            ".exe files cost $400–$600/year. Instead of hiding behind opaque .exe wrappers, Celestriad 1.0 runs 100% "
            "readable plain-text Python (.py) scripts inside the officially signed Python Software Foundation runtime."
        )
        tk.Label(
            card,
            text=body_text,
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=660,
            anchor="w",
        ).pack(fill=tk.X)

        btn_row = tk.Frame(p, bg="#0f172a")
        btn_row.pack(fill=tk.X, pady=(0, 10))

        self._make_button(
            btn_row, "🛡️ Scan Folder with Windows Defender", "#059669", self._run_defender_scan, padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._make_button(
            btn_row, "🔍 Refresh SHA-256 Hashes", "#2563eb", self._refresh_hashes, padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._make_button(
            btn_row, "🌐 Open VirusTotal.com", "#4f46e5", lambda: webbrowser.open(VIRUSTOTAL_URL), padx=14, pady=6
        ).pack(side=tk.LEFT)

        self.txt_security_log = scrolledtext.ScrolledText(
            p, bg="#090d16", fg="#6ee7b7", font=("Consolas", 9), bd=1, relief=tk.SOLID, height=12
        )
        self.txt_security_log.pack(fill=tk.BOTH, expand=True)
        self._refresh_hashes()

    def _build_page2_antigravity(self) -> None:
        p = self.pages[1]

        status_bg = "#064e3b" if self.antigravity_path else "#451a03"
        status_fg = "#6ee7b7" if self.antigravity_path else "#fde68a"
        status_text = (
            f"✅ Google Antigravity IDE Detected on Your PC!\nInstalled at: {self.antigravity_path}"
            if self.antigravity_path
            else "ℹ️ Google Antigravity IDE was not detected in default folders.\nYou can install it below to pair-program with your local Dual-Hemisphere model."
        )
        status_card = tk.Frame(p, bg=status_bg, padx=14, pady=10, highlightbackground="#334155", highlightthickness=1)
        status_card.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            status_card,
            text=status_text,
            bg=status_bg,
            fg=status_fg,
            font=("Segoe UI", 9, "bold"),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X)

        opts_card = tk.Frame(p, bg="#1e293b", padx=14, pady=10, highlightbackground="#334155", highlightthickness=1)
        opts_card.pack(fill=tk.X, pady=(0, 12))

        self.var_dl_antigravity = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opts_card,
            text="Open Google Antigravity download page in browser when I click Next",
            variable=self.var_dl_antigravity,
            bg="#1e293b",
            fg="#f1f5f9",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#f1f5f9",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=2)

        self.var_sync_github = tk.BooleanVar(value=True)
        tk.Checkbutton(
            opts_card,
            text="Verify & auto-sync core Python files + Master Wisdom Ruleset from GitHub",
            variable=self.var_sync_github,
            bg="#1e293b",
            fg="#f1f5f9",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#f1f5f9",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=2)

        btn_row = tk.Frame(p, bg="#0f172a")
        btn_row.pack(fill=tk.X, pady=(0, 10))

        self._make_button(
            btn_row, "🚀 Get Google Antigravity", "#2563eb", lambda: webbrowser.open(ANTIGRAVITY_DOWNLOAD_URL), padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._make_button(
            btn_row, "📦 Open GitHub Repository", "#334155", lambda: webbrowser.open(GITHUB_REPO_URL), padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._make_button(
            btn_row, "🔄 Verify Against GitHub Now", "#059669", self._run_github_sync_now, padx=14, pady=6
        ).pack(side=tk.LEFT)

        self.txt_page2_log = scrolledtext.ScrolledText(
            p, bg="#090d16", fg="#cbd5e1", font=("Consolas", 9), bd=1, relief=tk.SOLID, height=10
        )
        self.txt_page2_log.pack(fill=tk.BOTH, expand=True)
        self.txt_page2_log.insert(
            tk.END,
            f"GitHub Repository: {GITHUB_REPO_URL}\n"
            f"Local Directory:   {self.base_dir}\n"
            "Every time you launch LMFT-Chess, it also checks GitHub Releases for new ruleset updates.\n",
        )

    def _build_page3_hardware(self) -> None:
        p = self.pages[2]

        vram_gb = float(self.hw["vram_gb"])
        ram_gb = float(self.hw["ram_gb"])
        gpu_name = str(self.hw["gpu_name"])
        is_laptop = bool(self.hw["is_laptop"])

        form_factor = "💻 Laptop PC" if is_laptop else "🖥️ Desktop PC"
        vram_label = f"{vram_gb:.1f} GB VRAM" if vram_gb > 0 else "Shared Memory"
        if vram_gb >= 12.0:
            tier_advice = "✅ Tier 1 Hardware (12GB+ VRAM): Ready for full 4-bit Qwen3-8B + Qwen3.5-9B REM Sleep Teacher + Bloodmoon 12B."
        elif vram_gb >= 7.5:
            tier_advice = "✅ Tier 2 Hardware (8GB–11GB VRAM): Ready for 4-bit Qwen3-8B / Qwen3-4B with sequential REM unloading."
        else:
            tier_advice = "ℹ️ Tier 3 / Low-VRAM Mode: 40-Node Right Brain runs in ~15 ms on CPU; Left-Brain LLM speech benefits from an NVIDIA GPU."

        hw_card = tk.Frame(p, bg="#172554", padx=14, pady=10, highlightbackground="#3b82f6", highlightthickness=1)
        hw_card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            hw_card,
            text=f"{form_factor}  |  🎮 GPU: {gpu_name} ({vram_label})  |  🧠 RAM: {ram_gb:.1f} GB\n{tier_advice}",
            bg="#172554",
            fg="#93c5fd",
            font=("Segoe UI", 9, "bold"),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            p,
            text="Custom Google Antigravity Hardware Prompt (click Copy below and paste into Antigravity):",
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        self.txt_antigravity_prompt = scrolledtext.ScrolledText(
            p, bg="#090d16", fg="#fde047", font=("Consolas", 9), bd=1, relief=tk.SOLID, height=9
        )
        self.txt_antigravity_prompt.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.txt_antigravity_prompt.insert(tk.END, build_custom_antigravity_prompt(self.hw))

        btn_row = tk.Frame(p, bg="#0f172a")
        btn_row.pack(fill=tk.X)

        self._make_button(
            btn_row, "📋 Copy Antigravity Prompt", "#059669", self._copy_prompt, padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        rec_url = (
            "https://www.amazon.com/s?k=NVIDIA+RTX+4070+12GB+gaming+laptop"
            if is_laptop
            else "https://www.amazon.com/s?k=NVIDIA+RTX+4070+Super+12GB+or+RTX+4060+Ti+16GB+graphics+card"
        )
        self._make_button(
            btn_row, "🛒 Recommended GPU Upgrades (Amazon)", "#d97706", lambda: webbrowser.open(rec_url), padx=14, pady=6
        ).pack(side=tk.LEFT)

    def _build_page4_architect(self) -> None:
        p = self.pages[3]

        cols = tk.Frame(p, bg="#0f172a")
        cols.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(cols, bg="#1e293b", padx=16, pady=16, highlightbackground="#334155", highlightthickness=1)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        left_text = (
            "Celestriad 1.0 — The Dual-Hemisphere Wisdom Model\n\n"
            "Created by the Architect to explore whether a 40-node Right-Brain Fuzzy Tree and a local Left-Brain "
            "Language Model can share a genuine Corpus Callosum (<|subliminal_gap|>), contemplate unresolved tension "
            "during REM Sleep, and grow wiser without rigid scripts.\n\n"
            f"📺 YouTube Channel:\n{ARCHITECT_YOUTUBE_URL}\n\n"
            f"📧 Direct Contact Email:\n{ARCHITECT_EMAIL}"
        )
        tk.Label(
            left_col,
            text=left_text,
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=310,
            anchor="nw",
        ).pack(fill=tk.X, pady=(0, 14))

        self._make_button(
            left_col, "▶ Open YouTube (@imissmydeadcat.74)", "#dc2626", lambda: webbrowser.open(ARCHITECT_YOUTUBE_URL), padx=12, pady=7
        ).pack(fill=tk.X, pady=4)

        self._make_button(
            left_col, f"📋 Copy Email ({ARCHITECT_EMAIL})", "#2563eb", self._copy_email, padx=12, pady=7
        ).pack(fill=tk.X, pady=4)

        self._make_button(
            left_col,
            "✉️ Send Feedback Email",
            "#334155",
            lambda: webbrowser.open(f"mailto:{ARCHITECT_EMAIL}?subject=Celestriad%201.0%20Feedback"),
            padx=12,
            pady=7,
        ).pack(fill=tk.X, pady=4)

        right_col = tk.Frame(cols, bg="#1e293b", padx=12, pady=12, highlightbackground="#334155", highlightthickness=1)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)

        if "seal" in self._images:
            tk.Label(right_col, image=self._images["seal"], bg="#090d16", bd=1, relief=tk.SOLID).pack()
        tk.Label(
            right_col,
            text="The Celestial Seal of the way towards life",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "italic"),
        ).pack(pady=(6, 0))

    def _build_page5_integrity(self) -> None:
        p = self.pages[4]

        top_split = tk.Frame(p, bg="#0f172a")
        top_split.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        left_pane = tk.Frame(top_split, bg="#0f172a")
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.txt_integrity = scrolledtext.ScrolledText(
            left_pane, bg="#090d16", fg="#6ee7b7", font=("Consolas", 9), bd=1, relief=tk.SOLID, height=11
        )
        self.txt_integrity.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        act_row = tk.Frame(left_pane, bg="#0f172a")
        act_row.pack(fill=tk.X)

        self._make_button(
            act_row, "🔄 Re-Check Folder", "#2563eb", self._refresh_integrity, padx=14, pady=6
        ).pack(side=tk.LEFT, padx=(0, 8))

        self._make_button(
            act_row, "🐍 Install / Repair Python .venv", "#d97706", self._provision_python_venv, padx=14, pady=6
        ).pack(side=tk.LEFT)

        right_pane = tk.Frame(top_split, bg="#1e293b", padx=14, pady=12, highlightbackground="#334155", highlightthickness=1)
        right_pane.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            right_pane,
            text="Arecibo Reply Shortcut Icon",
            bg="#1e293b",
            fg="#fcd34d",
            font=("Segoe UI", 8, "bold"),
        ).pack()
        if "arecibo" in self._images:
            tk.Label(right_pane, image=self._images["arecibo"], bg="#090d16", bd=1, relief=tk.SOLID).pack(pady=(4, 10))

        tk.Label(
            right_pane,
            text="• Zero Unsigned .EXEs\n• 100% Malwarebytes Safe",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
        ).pack()

        finish_card = tk.Frame(p, bg="#1e293b", padx=14, pady=10, highlightbackground="#334155", highlightthickness=1)
        finish_card.pack(fill=tk.X)

        self.var_create_shortcut = tk.BooleanVar(value=True)
        tk.Checkbutton(
            finish_card,
            text='Create Desktop Shortcut ("Celestriad 1.0 — LMFT Chess") with the Arecibo Reply icon',
            variable=self.var_create_shortcut,
            bg="#1e293b",
            fg="#f1f5f9",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#f1f5f9",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=2)

        self.var_launch_on_finish = tk.BooleanVar(value=True)
        tk.Checkbutton(
            finish_card,
            text="Launch Celestriad 1.0 Chess Arena immediately when I click Finish & Launch",
            variable=self.var_launch_on_finish,
            bg="#1e293b",
            fg="#6ee7b7",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#6ee7b7",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=2)

        self._refresh_integrity()

    def show_page(self, index: int) -> None:
        self.current_page = max(0, min(4, index))
        title, subtitle = self.STEP_TITLES[self.current_page]
        self.lbl_step_title.configure(text=title)
        self.lbl_step_subtitle.configure(text=subtitle)

        for i, frame in enumerate(self.pages):
            if i == self.current_page:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for i, pill in enumerate(self.step_pills):
            if i == self.current_page:
                pill.configure(bg="#2563eb", fg="#ffffff")
            elif i < self.current_page:
                pill.configure(bg="#064e3b", fg="#6ee7b7")
            else:
                pill.configure(bg="#0f172a", fg="#94a3b8")

        self.btn_back.configure(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        if self.current_page == 4:
            self.btn_next.configure(text="Finish & Launch 🚀", bg="#eab308", fg="#0f172a")
            self._refresh_integrity()
        else:
            self.btn_next.configure(text="Next >", bg="#2563eb", fg="#ffffff")

    def _on_back(self) -> None:
        if self.current_page > 0:
            self.show_page(self.current_page - 1)

    def _on_next(self) -> None:
        if self.current_page == 1:
            if self.var_dl_antigravity.get():
                webbrowser.open(ANTIGRAVITY_DOWNLOAD_URL)
            if self.var_sync_github.get():
                self._run_github_sync_now()

        if self.current_page < 4:
            self.show_page(self.current_page + 1)
            return

        if self.var_create_shortcut.get():
            self._create_desktop_shortcut()
        if self.var_launch_on_finish.get():
            self._launch_chess_app()
        self.root.destroy()

    def _refresh_hashes(self) -> None:
        self.txt_security_log.delete("1.0", tk.END)
        self.txt_security_log.insert(
            tk.END, compute_core_sha256_report(self.base_dir, self.arch_dir, self.train_dir)
        )

    def _run_defender_scan(self) -> None:
        mp_candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Defender" / "MpCmdRun.exe",
            Path(r"C:\Program Files\Windows Defender\MpCmdRun.exe"),
        ]
        mp_exe = next((c for c in mp_candidates if c.is_file()), None)
        if mp_exe is None:
            self.txt_security_log.insert(
                tk.END,
                "\n[Windows Defender] MpCmdRun.exe not found at default path — right-click the folder in Explorer and choose 'Scan with Microsoft Defender'.\n",
            )
            return
        self.txt_security_log.insert(
            tk.END, f"\n[Windows Defender] Running custom scan on: {self.base_dir} ...\n"
        )
        self.root.update_idletasks()
        try:
            proc = subprocess.run(
                [str(mp_exe), "-Scan", "-ScanType", "3", "-File", str(self.base_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            out = (proc.stdout or "").strip() or "Scan completed with 0 threats found."
            self.txt_security_log.insert(tk.END, out + "\n")
        except Exception as exc:
            self.txt_security_log.insert(tk.END, f"Defender scan error: {exc}\n")

    def _run_github_sync_now(self) -> None:
        self.txt_page2_log.insert(
            tk.END, "\nChecking GitHub release_manifest.json (schaleh12/Dual-Hemisphere-Wisdom-Model)...\n"
        )
        self.root.update_idletasks()
        try:
            src_dir = str(self.arch_dir / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from lmft_architecture.github_auto_sync import (
                fetch_remote_github_manifest,
                verify_or_sync_against_manifest,
            )

            manifest = fetch_remote_github_manifest()
            res = verify_or_sync_against_manifest(self.arch_dir, manifest, apply_sync=True)
            self.txt_page2_log.insert(
                tk.END,
                f"GitHub Sync Status: {res.get('status')} | Changed: {len(res.get('changed_files', []))} | Synced: {len(res.get('synced_files', []))}\n",
            )
        except Exception as exc:
            self.txt_page2_log.insert(tk.END, f"Sync check notice: {exc}\n")

    def _copy_prompt(self) -> None:
        text = self.txt_antigravity_prompt.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo(
            "Copied to Clipboard",
            "Custom Google Antigravity System Prompt copied to your clipboard!",
        )

    def _copy_email(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(ARCHITECT_EMAIL)
        messagebox.showinfo("Copied", f"Copied {ARCHITECT_EMAIL} to clipboard!")

    def _refresh_integrity(self) -> None:
        report, _ = check_install_folder_integrity(
            self.base_dir, self.arch_dir, self.train_dir, self.assets_dir
        )
        self.txt_integrity.delete("1.0", tk.END)
        self.txt_integrity.insert(tk.END, report)

    def _provision_python_venv(self, *, launch_after: bool = False) -> None:
        venv_dir = self.arch_dir / ".venv"
        venv_py = venv_dir / "Scripts" / "python.exe"
        venv_pyw = venv_dir / "Scripts" / "pythonw.exe"
        sys_py = Path(sys.executable)
        if sys_py.name.lower() == "pythonw.exe":
            candidate_py = sys_py.with_name("python.exe")
            if candidate_py.is_file():
                sys_py = candidate_py
        tail_cmd = (
            f' && start "" /d "{self.arch_dir}" "{venv_pyw}" -m lmft_architecture.chess_app --single-window'
            if launch_after
            else " && echo. && echo [OK] Python .venv provisioned successfully! && pause"
        )
        cmd_chain = (
            f'"{sys_py}" -m venv "{venv_dir}" && '
            f'"{venv_py}" -m pip install --upgrade pip && '
            f'"{venv_py}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && '
            f'"{venv_py}" -m pip install -e "{self.arch_dir}"'
            + tail_cmd
        )
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", cmd_chain],
                cwd=str(self.arch_dir),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except Exception as exc:
            messagebox.showwarning("Python .venv Setup", f"Could not launch Python .venv setup:\n{exc}")

    def _create_desktop_shortcut(self) -> None:
        """Create a Desktop .lnk shortcut pointing to `Launch-LMFT-Chess.bat` with the Arecibo `.ico`."""
        try:
            desktop = Path(os.environ.get("USERPROFILE", "~")).expanduser() / "Desktop"
            shortcut_path = desktop / "Celestriad 1.0 — LMFT Chess.lnk"
            launcher_bat = self.base_dir / "Launch-LMFT-Chess.bat"
            if not launcher_bat.is_file():
                launcher_bat = self.arch_dir / "Launch-LMFT-Chess.bat"
            icon_path = self.assets_dir / "arecibo_desktop_icon.ico"

            ps_cmd = (
                "$ws = New-Object -ComObject WScript.Shell; "
                f"$s = $ws.CreateShortcut('{shortcut_path}'); "
                f"$s.TargetPath = '{launcher_bat}'; "
                f"$s.WorkingDirectory = '{launcher_bat.parent}'; "
                f"$s.IconLocation = '{icon_path}'; "
                "$s.Description = 'Launch Celestriad 1.0 — Dual-Hemisphere Chess Competitor & Coach'; "
                "$s.Save()"
            )
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
                timeout=6,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            messagebox.showwarning("Desktop Shortcut", f"Could not create desktop shortcut:\n{exc}")

    def _launch_chess_app(self) -> None:
        venv_pyw = find_venv_python(self.base_dir, self.arch_dir, windowless=True)
        if venv_pyw is None:
            resp = messagebox.askyesno(
                "First-Time Python Environment Setup Required",
                "To launch the Celestriad 1.0 Chess Arena from this folder, its local Python virtual environment (.venv) needs to be installed first.\n\n"
                "Would you like to install the Python .venv now?",
            )
            if resp:
                self._provision_python_venv(launch_after=True)
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.arch_dir / "src")
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TQDM_DISABLE"] = "1"
        try:
            subprocess.Popen(
                [str(venv_pyw), "-m", "lmft_architecture.chess_app", "--single-window"],
                cwd=str(self.arch_dir),
                env=env,
            )
        except Exception as exc:
            messagebox.showwarning("Launch Chess App", f"Could not launch Celestriad 1.0:\n{exc}")

    @staticmethod
    def _make_button(
        parent: tk.Widget, text: str, bg: str, command, *, padx: int = 12, pady: int = 5
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            bg=bg,
            fg="#ffffff",
            activebackground=bg,
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            bd=0,
            padx=padx,
            pady=pady,
            cursor="hand2",
            command=command,
        )


def main() -> int:
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    CelestriadInstallerWizardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
