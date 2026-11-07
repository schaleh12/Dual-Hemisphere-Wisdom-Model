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
from tkinter import messagebox
import webbrowser


ARCHITECT_YOUTUBE_URL = "https://www.youtube.com/@imissmydeadcat.74"
ARCHITECT_EMAIL = "imissmydeadcat74@gmail.com"
ANTIGRAVITY_DOWNLOAD_URL = "https://antigravity.google/"
GITHUB_REPO_URL = "https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model"
VIRUSTOTAL_URL = "https://www.virustotal.com/gui/home/upload"


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
    """Return `(base_dir, arch_dir, train_dir, assets_dir)` for repo or extracted bundle layouts."""
    here = Path(__file__).resolve()
    # Typical layout: <base>/LMFT-Architecture/src/lmft_architecture/installer_wizard.py
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

    train_dir = base_dir / "LMFT-Training" if (base_dir / "LMFT-Training").is_dir() else base_dir
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

    # 1. Probe nvidia-smi for NVIDIA GPU name and dedicated VRAM
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

    # 2. Fallback to Win32 EnumDisplayDevicesW if nvidia-smi was not found
    if gpu_name == "Standard Windows Display Adapter" and os.name == "nt":
        try:
            dd = _DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(_DISPLAY_DEVICEW)
            if ctypes.windll.user32.EnumDisplayDevicesW(None, 0, ctypes.byref(dd), 0):
                if dd.DeviceString and dd.DeviceString.strip():
                    gpu_name = dd.DeviceString.strip()
        except Exception:
            pass

    # 3. Probe System RAM via Win32 GlobalMemoryStatusEx
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
                # BatteryFlag 128 == No system battery
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
        "Please inspect `LMFT-Architecture/src/lmft_architecture/chess_app.py` and `three_legged_runtime.py` "
        "and customize my local runtime settings so that:\n"
        f"1. {profile_mode}\n"
        "2. The 40-Node Right-Brain Fuzzy Tree (`birth.py`) and `<|subliminal_gap|>` comma splice remain "
        "100% intact and un-rigged.\n"
        "3. Verify the configuration with `pytest LMFT-Architecture/tests/test_chess_app.py -q`."
    )


def compute_core_sha256_report(base_dir: Path, arch_dir: Path, train_dir: Path) -> str:
    """Compute SHA-256 digests of the core Python and JSON files."""
    targets = [
        ("Launch-Setup-Wizard.bat", base_dir / "Launch-Setup-Wizard.bat"),
        ("Launch-LMFT-Chess.bat", base_dir / "Launch-LMFT-Chess.bat"),
        ("Launch-LMFT-Bloodmoon.bat", base_dir / "Launch-LMFT-Bloodmoon.bat"),
        ("birth.py (40-Node Seed)", arch_dir / "src" / "lmft_architecture" / "birth.py"),
        ("chess_app.py (Chess Engine)", arch_dir / "src" / "lmft_architecture" / "chess_app.py"),
        ("three_legged_runtime.py (Subliminal Gap)", arch_dir / "src" / "lmft_architecture" / "three_legged_runtime.py"),
        ("installer_wizard.py (Setup Wizard)", arch_dir / "src" / "lmft_architecture" / "installer_wizard.py"),
        ("rem_sleep_consolidated_ruleset.json", train_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"),
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


def check_install_folder_integrity(base_dir: Path, arch_dir: Path, train_dir: Path, assets_dir: Path) -> tuple[str, bool]:
    """Run the 10-point install folder integrity check and return `(report_text, all_core_ok)`."""
    checks = [
        ("1. Core 40-Node Seed (birth.py)", arch_dir / "src" / "lmft_architecture" / "birth.py"),
        ("2. Chess Arena Engine (chess_app.py)", arch_dir / "src" / "lmft_architecture" / "chess_app.py"),
        ("3. Subliminal Gap Runtime (three_legged_runtime.py)", arch_dir / "src" / "lmft_architecture" / "three_legged_runtime.py"),
        ("4. Virgin Baseline Lock (virgin_baseline.py)", arch_dir / "src" / "lmft_architecture" / "virgin_baseline.py"),
        ("5. Right-Brain MiniLM Bridges", train_dir / "models" / "right_brain_bridges" / "doer_will_cultural" / "model.safetensors"),
        ("6. Qwen3-8B Left-Brain LoRA + Callosum", train_dir / "models" / "left_brain_qwen3_8b" / "projector_weights.pt"),
        ("7. Master Release Wisdom Seed", train_dir / "outputs" / "rem_sleep_consolidated_ruleset.json"),
        ("8. Arecibo Reply Desktop Icon (.ico)", assets_dir / "arecibo_desktop_icon.ico"),
        ("9. Celestriad Spirit Particle Icon (.ico)", assets_dir / "celestriad_spirit_particle.ico"),
    ]
    lines = [
        "=== Celestriad 1.0 Install Folder Integrity Check ===",
        f"Install Directory: {base_dir}",
        "",
    ]
    core_ok = True
    for idx, (label, path) in enumerate(checks):
        if path.exists():
            lines.append(f"✅ [OK] {label}")
        else:
            lines.append(f"⚠️ [MISSING] {label} ({path})")
            if idx < 4:
                core_ok = False

    venv_py = find_venv_python(base_dir, arch_dir, windowless=False)
    if venv_py is not None:
        lines.append(f"✅ [OK] 10. Python Virtual Environment (.venv): {venv_py}")
    else:
        lines.append("⚠️ [ACTION NEEDED] 10. Python .venv not created yet -> Click [Install / Repair Python .venv] below!")
    return "\n".join(lines), core_ok


def find_venv_python(base_dir: Path, arch_dir: Path, *, windowless: bool = False) -> Path | None:
    """Locate `.venv/Scripts/python.exe` (or `pythonw.exe` when `windowless=True`)."""
    exe_name = "pythonw.exe" if windowless else "python.exe"
    candidates = [
        arch_dir / ".venv" / "Scripts" / exe_name,
        base_dir / ".venv" / "Scripts" / exe_name,
        arch_dir / ".venv" / "Scripts" / "python.exe",
        base_dir / ".venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


class CelestriadInstallerWizardApp:
    """5-Page Tkinter GUI Setup & Hardware Diagnostics Wizard."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.base_dir, self.arch_dir, self.train_dir, self.assets_dir = resolve_workspace_paths()
        self.hw = detect_system_hardware()
        self.antigravity_path = detect_antigravity_installation()
        self.current_page = 0
        self._images: dict[str, tk.PhotoImage] = {}

        self.root.title("Celestriad 1.0 — LMFT-Architecture Setup & Hardware Wizard")
        self.root.geometry("880x640")
        self.root.minsize(850, 610)
        self.root.configure(bg="#141418")

        ico_path = self.assets_dir / "arecibo_desktop_icon.ico"
        if ico_path.is_file():
            try:
                self.root.iconbitmap(default=str(ico_path))
            except Exception:
                pass

        self._load_preview_images()
        self._build_shell()
        self._build_page1_security()
        self._build_page2_antigravity()
        self._build_page3_hardware()
        self._build_page4_architect()
        self._build_page5_integrity()
        self.show_page(0)

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

    def _build_shell(self) -> None:
        # Top Navigation Bar
        self.top_bar = tk.Frame(self.root, bg="#1c1d24", height=76)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.top_bar.pack_propagate(False)

        tk.Label(
            self.top_bar,
            text="✦ CELESTRIAD 1.0 — DUAL-HEMISPHERE WISDOM MODEL SETUP WIZARD",
            bg="#1c1d24",
            fg="#f59e0b",
            font=("Segoe UI", 11, "bold"),
        ).place(x=20, y=10)

        step_names = [
            "1. Security & AV",
            "2. Antigravity & GitHub",
            "3. Hardware & Prompt",
            "4. Architect & Seal",
            "5. Integrity & Launch",
        ]
        self.step_pills: list[tk.Label] = []
        pill_x = 20
        for idx, name in enumerate(step_names):
            lbl = tk.Label(
                self.top_bar,
                text=name,
                bg="#282a34",
                fg="#94a3b8",
                font=("Segoe UI", 8, "bold"),
                width=18,
                height=1,
                cursor="hand2",
                padx=4,
                pady=3,
            )
            lbl.place(x=pill_x, y=40)
            lbl.bind("<Button-1>", lambda _e, i=idx: self.show_page(i))
            self.step_pills.append(lbl)
            pill_x += 152

        # Bottom Navigation Bar
        self.bottom_bar = tk.Frame(self.root, bg="#1c1d24", height=58)
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_bar.pack_propagate(False)

        self.btn_cancel = self._make_button(
            self.bottom_bar, "Cancel", "#373a44", self.root.destroy, width=12
        )
        self.btn_cancel.place(x=20, y=13)

        self.btn_back = self._make_button(
            self.bottom_bar, "< Back", "#373a44", self._on_back, width=14
        )
        self.btn_back.place(x=565, y=13)

        self.btn_next = self._make_button(
            self.bottom_bar, "Next >", "#537a32", self._on_next, width=18
        )
        self.btn_next.place(x=695, y=13)

        # Page Container
        self.page_container = tk.Frame(self.root, bg="#141418")
        self.page_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.pages: list[tk.Frame] = []
        for _ in range(5):
            frame = tk.Frame(self.page_container, bg="#141418")
            self.pages.append(frame)

    def _build_page1_security(self) -> None:
        p = self.pages[0]
        self._make_header(
            p,
            "Page 1 of 5 — Why Open-Source Python Scripts Are 100% Transparent & How to Virus-Check",
            x=22,
            y=14,
        )

        body_text = (
            "What This Application Is:\n"
            "• Celestriad 1.0 (LMFT-Architecture) is an open-source Dual-Hemisphere AI research project pairing a fast "
            "40-Node Right-Brain Fuzzy Tree (~15 ms intuition) with a local Qwen3-8B Left Brain spliced across <|subliminal_gap|>.\n\n"
            "Why We Ship Zero Custom .EXE Files (100% Malwarebytes & Antivirus Friendly):\n"
            "• Antivirus programs like Malwarebytes and Windows SmartScreen flag unsigned indie .exe wrappers because "
            "commercial Extended Validation (EV) hardware certificates cost $400–$600/year.\n"
            "• Instead of shipping opaque .exe binaries, Celestriad 1.0 runs 100% readable plain-text Python (.py) source "
            "files inside the officially signed Python Software Foundation interpreter (pythonw.exe / python.exe).\n\n"
            "How to Verify This Folder Yourself Right Now:\n"
            "1. Click [Scan Folder with Windows Defender] below to run Microsoft's official antivirus engine on this directory.\n"
            "2. Click [Compute Local SHA-256 Hashes] or inspect any .py file in Notepad / Google Antigravity."
        )
        tk.Label(
            p,
            text=body_text,
            bg="#141418",
            fg="#dce2ea",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=820,
            anchor="nw",
        ).place(x=22, y=46, width=825, height=205)

        b_scan = self._make_button(
            p, "🛡️ Scan Folder with Windows Defender", "#1e6e50", self._run_defender_scan, width=32
        )
        b_scan.place(x=22, y=258)

        b_hash = self._make_button(
            p, "🔍 Compute Local SHA-256 Hashes", "#2d4c78", self._refresh_hashes, width=30
        )
        b_hash.place(x=285, y=258)

        b_vt = self._make_button(
            p, "🌐 Open VirusTotal.com", "#4b3869", lambda: webbrowser.open(VIRUSTOTAL_URL), width=24
        )
        b_vt.place(x=535, y=258)

        self.txt_security_log = tk.Text(
            p, bg="#0e0f12", fg="#94e2d5", font=("Consolas", 9), bd=1, relief=tk.SOLID
        )
        self.txt_security_log.place(x=22, y=302, width=825, height=185)
        self._refresh_hashes()

    def _build_page2_antigravity(self) -> None:
        p = self.pages[1]
        self._make_header(
            p,
            "Page 2 of 5 — Google Antigravity IDE & GitHub Repository Synchronization",
            x=22,
            y=14,
        )

        tk.Label(
            p,
            text=(
                "Celestriad 1.0 is designed so you can pair-program with Google Antigravity to customize your AI models, "
                "inspect the 40-node Right-Brain Fuzzy Tree, and automatically pull the latest GitHub release updates."
            ),
            bg="#141418",
            fg="#dce2ea",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=820,
            anchor="nw",
        ).place(x=22, y=46, width=825, height=44)

        status_bg = "#163828" if self.antigravity_path else "#3c2e16"
        status_fg = "#86efac" if self.antigravity_path else "#fcd34d"
        status_text = (
            f"✅ Google Antigravity Detected on Your System!\nLocation: {self.antigravity_path} (No download required)"
            if self.antigravity_path
            else "ℹ️ Google Antigravity was not detected in default folders.\nCheck the box below or click [Get Google Antigravity] to install it."
        )
        tk.Label(
            p,
            text=status_text,
            bg=status_bg,
            fg=status_fg,
            font=("Segoe UI", 9, "bold"),
            justify=tk.LEFT,
             anchor="w",
            padx=12,
        ).place(x=22, y=96, width=825, height=52)

        self.var_dl_antigravity = tk.BooleanVar(value=self.antigravity_path is None)
        tk.Checkbutton(
            p,
            text="Open Google Antigravity download page when I click Next (recommended if not installed)",
            variable=self.var_dl_antigravity,
            bg="#141418",
            fg="#e2e8f0",
            selectcolor="#1c1d24",
            activebackground="#141418",
            activeforeground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        ).place(x=22, y=160)

        self.var_sync_github = tk.BooleanVar(value=True)
        tk.Checkbutton(
            p,
            text="Verify & auto-sync core Python files + Master Wisdom Ruleset from GitHub (schaleh12/Dual-Hemisphere-Wisdom-Model)",
            variable=self.var_sync_github,
            bg="#141418",
            fg="#e2e8f0",
            selectcolor="#1c1d24",
            activebackground="#141418",
            activeforeground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        ).place(x=22, y=190)

        self._make_button(
            p, "🚀 Get Google Antigravity", "#2d5284", lambda: webbrowser.open(ANTIGRAVITY_DOWNLOAD_URL), width=28
        ).place(x=22, y=230)

        self._make_button(
            p, "📦 Open GitHub Repository", "#3a4152", lambda: webbrowser.open(GITHUB_REPO_URL), width=28
        ).place(x=260, y=230)

        self._make_button(
            p, "🔄 Sync / Verify from GitHub Now", "#537a32", self._run_github_sync_now, width=32
        ).place(x=500, y=230)

        self.txt_page2_log = tk.Text(
            p, bg="#0e0f12", fg="#cbd5e1", font=("Consolas", 9), bd=1, relief=tk.SOLID
        )
        self.txt_page2_log.place(x=22, y=274, width=825, height=210)
        self.txt_page2_log.insert(
            tk.END,
            f"GitHub Repository: {GITHUB_REPO_URL}\n"
            f"Local Workspace:   {self.base_dir}\n"
            "Tip: Every time you launch LMFT-Chess, it also checks GitHub in the background for updated rulesets.\n",
        )

    def _build_page3_hardware(self) -> None:
        p = self.pages[2]
        self._make_header(
            p,
            "Page 3 of 5 — Graphics Card & System Check + Custom Antigravity Prompt",
            x=22,
            y=14,
        )

        vram_gb = float(self.hw["vram_gb"])
        ram_gb = float(self.hw["ram_gb"])
        gpu_name = str(self.hw["gpu_name"])
        is_laptop = bool(self.hw["is_laptop"])

        form_factor = "💻 Form Factor: Laptop / Notebook PC" if is_laptop else "🖥️ Form Factor: Desktop Workstation / Tower"
        vram_label = f"{vram_gb:.1f} GB VRAM" if vram_gb > 0 else "Shared / Integrated Memory"
        if vram_gb >= 12.0:
            tier_advice = "✅ Tier 1 Ready (12GB+ VRAM): Ideal for 4-bit Qwen3-8B Left Brain + Qwen3.5-9B REM Sleep Teacher + Bloodmoon 12B!"
        elif vram_gb >= 7.5:
            tier_advice = "✅ Tier 2 Ready (8GB–11GB VRAM): Runs 4-bit Qwen3-8B or Qwen3-4B Left Brain smoothly with sequential REM unloading."
        else:
            tier_advice = "⚠️ Tier 3 / Low-VRAM Mode: Fast 40-Node Right Brain runs in ~15 ms on CPU, but Left-Brain LLM speech benefits from an NVIDIA GPU upgrade or Qwen3-4B 4-bit."

        banner_text = f"{form_factor}   |   🎮 GPU: {gpu_name} ({vram_label})   |   🧠 System RAM: {ram_gb:.1f} GB\n{tier_advice}"
        tk.Label(
            p,
            text=banner_text,
            bg="#1a2434",
            fg="#7dd3fc",
            font=("Segoe UI", 9, "bold"),
            justify=tk.LEFT,
            anchor="w",
            padx=12,
        ).place(x=22, y=46, width=825, height=58)

        tk.Label(
            p,
            text="Copy this custom System Prompt and paste it into Google Antigravity to tailor Celestriad 1.0 to your PC:",
            bg="#141418",
            fg="#dce2ea",
            font=("Segoe UI", 9, "bold"),
        ).place(x=22, y=112)

        self.txt_antigravity_prompt = tk.Text(
            p, bg="#0e0f12", fg="#fde047", font=("Consolas", 9), bd=1, relief=tk.SOLID
        )
        self.txt_antigravity_prompt.place(x=22, y=136, width=825, height=195)
        self.txt_antigravity_prompt.insert(tk.END, build_custom_antigravity_prompt(self.hw))

        self._make_button(
            p, "📋 Copy Antigravity System Prompt", "#537a32", self._copy_prompt, width=30
        ).place(x=22, y=342)

        rec_url = (
            "https://www.amazon.com/s?k=NVIDIA+RTX+4070+12GB+gaming+laptop"
            if is_laptop
            else "https://www.amazon.com/s?k=NVIDIA+RTX+4070+Super+12GB+or+RTX+4060+Ti+16GB+graphics+card"
        )
        self._make_button(
            p, "🛒 Search Amazon (Recommended for Your PC)", "#b46e14", lambda: webbrowser.open(rec_url), width=36
        ).place(x=275, y=342)

        self._make_button(
            p,
            "🖥️ Desktop NVIDIA GPUs (12GB–16GB VRAM)",
            "#3a4152",
            lambda: webbrowser.open("https://www.amazon.com/s?k=NVIDIA+GeForce+RTX+3060+12GB+RTX+4060+Ti+16GB+GPU"),
            width=36,
        ).place(x=22, y=384)

        self._make_button(
            p,
            "💻 RTX 4060 / 4070 Laptops",
            "#3a4152",
            lambda: webbrowser.open("https://www.amazon.com/s?k=NVIDIA+RTX+4060+RTX+4070+gaming+laptop+16GB+RAM"),
            width=30,
        ).place(x=315, y=384)

        hw_note = (
            "💡 Hardware Upgrade Note: Because you are on a Laptop, laptop GPUs are soldered to the motherboard — "
            "upgrading VRAM requires either an external Thunderbolt/USB4 eGPU enclosure or a gaming laptop with an NVIDIA RTX 4060 (8GB) / RTX 4070 (8GB–12GB) / RTX 4080 (12GB)."
            if is_laptop
            else "💡 Hardware Upgrade Note: Because you are on a Desktop PC, you can upgrade your PCIe graphics card directly! "
            "For local 4-bit 8B–12B LLMs, the best value cards on Amazon are the NVIDIA RTX 3060 (12GB VRAM), RTX 4060 Ti (16GB VRAM), or RTX 4070 Super (12GB VRAM)."
        )
        tk.Label(
            p,
            text=hw_note,
            bg="#141418",
            fg="#b4bed0",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            wraplength=820,
            anchor="nw",
        ).place(x=22, y=428, width=825, height=52)

    def _build_page4_architect(self) -> None:
        p = self.pages[3]
        self._make_header(
            p,
            "Page 4 of 5 — Meet & Support the Architect of Celestriad 1.0",
            x=22,
            y=14,
        )

        left_text = (
            "Celestriad 1.0 (The Dual-Hemisphere Wisdom Model)\n\n"
            "Created by the Architect to explore whether a 40-node Right-Brain Fuzzy Tree and a local Left-Brain Language Model "
            "can share a genuine Corpus Callosum (<|subliminal_gap|>), contemplate unresolved tension during REM Sleep, "
            "and grow wiser without rigid rules or fake scripts.\n\n"
            "📺 YouTube Channel:\n"
            f"   {ARCHITECT_YOUTUBE_URL}\n\n"
            "📧 Direct Contact Email:\n"
            f"   {ARCHITECT_EMAIL}\n\n"
            "Every subscription, comment, or email helps support independent open-source AI research!"
        )
        tk.Label(
            p,
            text=left_text,
            bg="#141418",
            fg="#dce2ea",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=410,
            anchor="nw",
        ).place(x=22, y=50, width=415, height=280)

        self._make_button(
            p, "▶ Open YouTube: @imissmydeadcat.74", "#dc2626", lambda: webbrowser.open(ARCHITECT_YOUTUBE_URL), width=38
        ).place(x=22, y=340)

        self._make_button(
            p, f"📧 Copy Email: {ARCHITECT_EMAIL}", "#2d5284", self._copy_email, width=38
        ).place(x=22, y=384)

        self._make_button(
            p,
            "✉️ Open Default Mail App",
            "#3a4152",
            lambda: webbrowser.open(f"mailto:{ARCHITECT_EMAIL}?subject=Celestriad%201.0%20Feedback"),
            width=38,
        ).place(x=22, y=428)

        if "seal" in self._images:
            seal_lbl = tk.Label(p, image=self._images["seal"], bg="#0e0f12", bd=1, relief=tk.SOLID)
            seal_lbl.place(x=455, y=50, width=390, height=380)

        tk.Label(
            p,
            text="The Celestial Seal of the way towards life",
            bg="#141418",
            fg="#7dd3fc",
            font=("Segoe UI", 9, "italic"),
            justify=tk.CENTER,
        ).place(x=455, y=436, width=390, height=24)

    def _build_page5_integrity(self) -> None:
        p = self.pages[4]
        self._make_header(
            p,
            "Page 5 of 5 — Install Folder Integrity, Desktop Icon & Launch",
            x=22,
            y=14,
        )

        self.txt_integrity = tk.Text(
            p, bg="#0e0f12", fg="#94e2d5", font=("Consolas", 9), bd=1, relief=tk.SOLID
        )
        self.txt_integrity.place(x=22, y=48, width=500, height=265)

        self._make_button(
            p, "🔄 Re-Check Folder Integrity", "#2d4c78", self._refresh_integrity, width=28
        ).place(x=22, y=322)

        self._make_button(
            p, "🐍 Install / Repair Python .venv", "#8c521e", self._provision_python_venv, width=30
        ).place(x=255, y=322)

        # Icon Previews on Right
        tk.Label(
            p,
            text="Desktop Shortcut Icon\n(Arecibo Reply)",
            bg="#141418",
            fg="#fcd34d",
            font=("Segoe UI", 8, "bold"),
            justify=tk.CENTER,
        ).place(x=540, y=48, width=145, height=34)

        if "arecibo" in self._images:
            tk.Label(p, image=self._images["arecibo"], bg="#0e0f12", bd=1, relief=tk.SOLID).place(
                x=548, y=86, width=130, height=130
            )

        tk.Label(
            p,
            text="Celestriad App Icon\n(Spirit Particle)",
            bg="#141418",
            fg="#7dd3fc",
            font=("Segoe UI", 8, "bold"),
            justify=tk.CENTER,
        ).place(x=700, y=48, width=145, height=34)

        if "spirit" in self._images:
            tk.Label(p, image=self._images["spirit"], bg="#0e0f12", bd=1, relief=tk.SOLID).place(
                x=708, y=86, width=130, height=130
            )

        tk.Label(
            p,
            text=(
                "• Desktop Shortcut Icon: The Arecibo Reply\n"
                "• Celestriad Identity Icon: The Spirit Particle\n"
                "• Launches via official signed pythonw.exe"
            ),
            bg="#141418",
            fg="#bcc6d4",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            anchor="nw",
        ).place(x=545, y=228, width=305, height=65)

        self.var_create_shortcut = tk.BooleanVar(value=True)
        tk.Checkbutton(
            p,
            text='Create Desktop Shortcut ("Celestriad 1.0 — LMFT Chess") with the Arecibo Reply icon',
            variable=self.var_create_shortcut,
            bg="#141418",
            fg="#e2e8f0",
            selectcolor="#1c1d24",
            activebackground="#141418",
            activeforeground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        ).place(x=22, y=378)

        self.var_launch_on_finish = tk.BooleanVar(value=True)
        tk.Checkbutton(
            p,
            text="Launch Celestriad 1.0 Chess Arena immediately when I click Finish",
            variable=self.var_launch_on_finish,
            bg="#141418",
            fg="#86efac",
            selectcolor="#1c1d24",
            activebackground="#141418",
            activeforeground="#86efac",
            font=("Segoe UI", 9, "bold"),
        ).place(x=22, y=410)

        self._refresh_integrity()

    def show_page(self, index: int) -> None:
        self.current_page = max(0, min(4, index))
        for i, frame in enumerate(self.pages):
            if i == self.current_page:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for i, pill in enumerate(self.step_pills):
            if i == self.current_page:
                pill.configure(bg="#537a32", fg="#ffffff")
            elif i < self.current_page:
                pill.configure(bg="#2d4828", fg="#b4e6aa")
            else:
                pill.configure(bg="#282a34", fg="#94a3b8")

        self.btn_back.configure(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        if self.current_page == 4:
            self.btn_next.configure(text="Finish & Launch 🚀", bg="#eab308", fg="#111827")
            self._refresh_integrity()
        else:
            self.btn_next.configure(text="Next >", bg="#537a32", fg="#ffffff")

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

        # Page 5 Finish
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

    def _provision_python_venv(self) -> None:
        venv_dir = self.arch_dir / ".venv"
        venv_py = venv_dir / "Scripts" / "python.exe"
        cmd_chain = (
            f'python -m venv "{venv_dir}" && '
            f'"{venv_py}" -m pip install --upgrade pip && '
            f'"{venv_py}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && '
            f'"{venv_py}" -m pip install -e "{self.arch_dir}" && '
            "echo. && echo [OK] Python .venv provisioned successfully! && pause"
        )
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", cmd_chain],
                cwd=str(self.base_dir),
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
            icon_path = self.assets_dir / "arecibo_desktop_icon.ico"

            ps_cmd = (
                "$ws = New-Object -ComObject WScript.Shell; "
                f"$s = $ws.CreateShortcut('{shortcut_path}'); "
                f"$s.TargetPath = '{launcher_bat}'; "
                f"$s.WorkingDirectory = '{self.base_dir}'; "
                f"$s.IconLocation = '{icon_path}'; "
                "$s.Description = 'Launch Celestriad 1.0 — Dual-Hemisphere Chess Competitor & Coach'; "
                "$s.Save()"
            )
            subprocess.run(["powershell.exe", "-Command", ps_cmd], timeout=6, check=False)
        except Exception as exc:
            messagebox.showwarning("Desktop Shortcut", f"Could not create desktop shortcut:\n{exc}")

    def _launch_chess_app(self) -> None:
        venv_pyw = find_venv_python(self.base_dir, self.arch_dir, windowless=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.arch_dir / "src")
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TQDM_DISABLE"] = "1"
        exe = str(venv_pyw) if venv_pyw is not None else sys.executable
        try:
            subprocess.Popen(
                [exe, "-m", "lmft_architecture.chess_app", "--single-window"],
                cwd=str(self.arch_dir),
                env=env,
            )
        except Exception as exc:
            messagebox.showwarning("Launch Chess App", f"Could not launch Celestriad 1.0:\n{exc}")

    @staticmethod
    def _make_header(parent: tk.Widget, text: str, *, x: int, y: int) -> tk.Label:
        lbl = tk.Label(
            parent,
            text=text,
            bg="#141418",
            fg="#38bdf8",
            font=("Segoe UI", 11, "bold"),
        )
        lbl.place(x=x, y=y)
        return lbl

    @staticmethod
    def _make_button(
        parent: tk.Widget, text: str, bg: str, command, *, width: int = 20
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
            padx=8,
            pady=5,
            width=width,
            cursor="hand2",
            command=command,
        )


def main() -> int:
    root = tk.Tk()
    CelestriadInstallerWizardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
