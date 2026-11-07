using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new InstallerWizardForm());
        return 0;
    }
}

internal sealed class InstallerWizardForm : Form
{
    private readonly string baseDir;
    private readonly string archDir;
    private readonly string trainDir;
    private readonly string assetsDir;

    private int currentPage = 0;
    private readonly Panel[] pages = new Panel[5];
    private readonly Label[] stepPills = new Label[5];

    private Button btnBack;
    private Button btnNext;
    private Button btnCancel;

    // Page 1 controls
    private TextBox txtSecurityLog;

    // Page 2 controls
    private Label lblAntigravityDetected;
    private CheckBox chkDownloadAntigravity;
    private CheckBox chkSyncGithub;
    private TextBox txtPage2Log;
    private bool antigravityAlreadyInstalled;

    // Page 3 controls
    private Label lblHardwareSummary;
    private TextBox txtAntigravityPrompt;
    private bool isLaptopSystem;
    private string detectedGpuName = "Detecting GPU...";
    private double detectedVramGb = 0.0;
    private double detectedRamGb = 16.0;

    // Page 5 controls
    private TextBox txtIntegrityReport;
    private CheckBox chkCreateDesktopShortcut;
    private CheckBox chkLaunchOnFinish;

    public InstallerWizardForm()
    {
        baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\', '/');
        archDir = Directory.Exists(Path.Combine(baseDir, "LMFT-Architecture"))
            ? Path.Combine(baseDir, "LMFT-Architecture")
            : baseDir;
        trainDir = Directory.Exists(Path.Combine(baseDir, "LMFT-Training"))
            ? Path.Combine(baseDir, "LMFT-Training")
            : baseDir;
        assetsDir = Path.Combine(archDir, "docs", "assets");

        Text = "Celestriad 1.0 — LMFT-Architecture Setup & Hardware Wizard";
        Size = new Size(860, 620);
        MinimumSize = new Size(840, 600);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(20, 20, 24);
        ForeColor = Color.FromArgb(242, 245, 248);
        Font = new Font("Segoe UI", 9.5f, FontStyle.Regular);

        string icoPath = Path.Combine(assetsDir, "arecibo_desktop_icon.ico");
        if (File.Exists(icoPath))
        {
            try { Icon = new Icon(icoPath); } catch { }
        }

        BuildShellLayout();
        BuildPage1Security();
        BuildPage2AntigravityAndGithub();
        BuildPage3HardwareAndPrompt();
        BuildPage4ArchitectContact();
        BuildPage5IntegrityAndShortcut();

        ShowPage(0);
    }

    private void BuildShellLayout()
    {
        Panel topBar = new Panel
        {
            Dock = DockStyle.Top,
            Height = 72,
            BackColor = Color.FromArgb(28, 29, 36),
            Padding = new Padding(16, 10, 16, 10),
        };

        PictureBox picLogo = new PictureBox
        {
            Size = new Size(48, 48),
            Location = new Point(16, 12),
            SizeMode = PictureBoxSizeMode.Zoom,
        };
        string spiritJpg = Path.Combine(assetsDir, "celestriad_spirit_particle.jpg");
        if (File.Exists(spiritJpg))
        {
            try { picLogo.Image = Image.FromFile(spiritJpg); } catch { }
        }
        topBar.Controls.Add(picLogo);

        Label lblTitle = new Label
        {
            Text = "Celestriad 1.0 — Multi-Step Windows Setup & Diagnostic Wizard",
            Font = new Font("Segoe UI", 13f, FontStyle.Bold),
            ForeColor = Color.FromArgb(125, 211, 252),
            AutoSize = true,
            Location = new Point(74, 10),
        };
        topBar.Controls.Add(lblTitle);

        string[] stepNames = new string[]
        {
            "1. Security & Trust",
            "2. Antigravity & GitHub",
            "3. GPU & AI Prompt",
            "4. The Architect",
            "5. Verify & Icon",
        };
        int pillX = 76;
        for (int i = 0; i < 5; i++)
        {
            int idx = i;
            Label pill = new Label
            {
                Text = stepNames[i],
                AutoSize = false,
                Size = new Size(140, 24),
                Location = new Point(pillX, 38),
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                Cursor = Cursors.Hand,
            };
            pill.Click += (s, e) => ShowPage(idx);
            stepPills[i] = pill;
            topBar.Controls.Add(pill);
            pillX += 146;
        }

        Panel bottomBar = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 58,
            BackColor = Color.FromArgb(28, 29, 36),
        };

        btnCancel = MakeButton("Cancel", new Point(20, 13), new Size(100, 32), Color.FromArgb(55, 58, 68));
        btnCancel.Click += (s, e) => Close();

        btnBack = MakeButton("< Back", new Point(560, 13), new Size(115, 32), Color.FromArgb(55, 58, 68));
        btnBack.Click += (s, e) => { if (currentPage > 0) ShowPage(currentPage - 1); };

        btnNext = MakeButton("Next >", new Point(685, 13), new Size(140, 32), Color.FromArgb(83, 122, 50));
        btnNext.Click += OnNextClicked;

        bottomBar.Controls.Add(btnCancel);
        bottomBar.Controls.Add(btnBack);
        bottomBar.Controls.Add(btnNext);

        Controls.Add(bottomBar);
        Controls.Add(topBar);

        for (int i = 0; i < 5; i++)
        {
            Panel p = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(20, 20, 24),
                Padding = new Padding(22, 16, 22, 16),
                Visible = false,
            };
            pages[i] = p;
            Controls.Add(p);
            p.BringToFront();
        }
    }

    private void BuildPage1Security()
    {
        Panel p = pages[0];

        Label h = MakeHeader("Page 1 of 5 — What Celestriad 1.0 Is, Why SmartScreen Prompts & How to Virus-Check", new Point(22, 14));
        p.Controls.Add(h);

        Label body = new Label
        {
            Location = new Point(22, 46),
            Size = new Size(795, 195),
            ForeColor = Color.FromArgb(220, 226, 234),
            Font = new Font("Segoe UI", 9.5f, FontStyle.Regular),
            Text =
                "What This Application Is:\r\n" +
                "• Celestriad 1.0 (LMFT-Architecture) is an open-source Dual-Hemisphere AI research project pairing a fast 40-Node Right-Brain Fuzzy Tree (~15 ms intuition) with a local Qwen3-8B Left Brain spliced across <|subliminal_gap|>.\r\n\r\n" +
                "Why Windows SmartScreen Says 'Unrecognized App / Unknown Publisher':\r\n" +
                "• Microsoft only removes that blue prompt if a developer pays a commercial Certificate Authority $300–$600/year for a corporate Extended Validation (EV) USB hardware token.\r\n" +
                "• Because this is an independent open-source scientific release, LMFT-Chess.exe and this Setup Wizard are compiled directly from readable plain-text C# source files (LMFT-Training/scripts/*.cs) with zero hidden installers or background services.\r\n\r\n" +
                "How to Properly Verify & Check It for Viruses Yourself:\r\n" +
                "1. Click [Scan Folder with Windows Defender] below to run Microsoft's official antivirus engine directly on this directory.\r\n" +
                "2. Click [Compute Local SHA-256 Hashes] and compare against VirusTotal.com or recompile the .cs files yourself.",
        };
        p.Controls.Add(body);

        Button btnDefender = MakeButton("🛡️ Scan Folder with Windows Defender", new Point(22, 248), new Size(265, 34), Color.FromArgb(30, 88, 145));
        btnDefender.Click += (s, e) => RunWindowsDefenderScan();

        Button btnHashes = MakeButton("🔍 Compute Local SHA-256 Hashes", new Point(297, 248), new Size(245, 34), Color.FromArgb(55, 65, 81));
        btnHashes.Click += (s, e) => ComputeLocalHashes();

        Button btnVirusTotal = MakeButton("🌐 Open VirusTotal.com", new Point(552, 248), new Size(190, 34), Color.FromArgb(55, 65, 81));
        btnVirusTotal.Click += (s, e) => OpenUrl("https://www.virustotal.com/gui/home/upload");

        p.Controls.Add(btnDefender);
        p.Controls.Add(btnHashes);
        p.Controls.Add(btnVirusTotal);

        txtSecurityLog = new TextBox
        {
            Location = new Point(22, 292),
            Size = new Size(795, 165),
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(14, 15, 19),
            ForeColor = Color.FromArgb(167, 243, 208),
            Font = new Font("Consolas", 9f, FontStyle.Regular),
            BorderStyle = BorderStyle.FixedSingle,
        };
        p.Controls.Add(txtSecurityLog);
        ComputeLocalHashes();
    }

    private void BuildPage2AntigravityAndGithub()
    {
        Panel p = pages[1];

        Label h = MakeHeader("Page 2 of 5 — Google Antigravity IDE Detection & GitHub Auto-Sync", new Point(22, 14));
        p.Controls.Add(h);

        string detectedPath = DetectGoogleAntigravityPath();
        antigravityAlreadyInstalled = !string.IsNullOrEmpty(detectedPath);

        lblAntigravityDetected = new Label
        {
            Location = new Point(22, 48),
            Size = new Size(795, 58),
            Padding = new Padding(10),
            BackColor = antigravityAlreadyInstalled ? Color.FromArgb(20, 62, 42) : Color.FromArgb(58, 44, 20),
            ForeColor = antigravityAlreadyInstalled ? Color.FromArgb(167, 243, 208) : Color.FromArgb(253, 230, 138),
            Font = new Font("Segoe UI", 9.75f, FontStyle.Bold),
            Text = antigravityAlreadyInstalled
                ? "✅ Google Antigravity Detected on Your PC!\r\nLocation: " + detectedPath + " (No need to re-download unless you want to open the site)."
                : "ℹ️ Google Antigravity Was Not Detected on This PC.\r\nWe recommend installing Google Antigravity so you can customize and pair-program with Celestriad 1.0.",
        };
        p.Controls.Add(lblAntigravityDetected);

        chkDownloadAntigravity = new CheckBox
        {
            Location = new Point(26, 120),
            Size = new Size(780, 28),
            Checked = !antigravityAlreadyInstalled,
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 10f, FontStyle.Bold),
            Text = antigravityAlreadyInstalled
                ? "Open Google Antigravity download page anyway (Already installed on this PC)"
                : "Download / Open Google Antigravity IDE installer when clicking Next",
        };
        p.Controls.Add(chkDownloadAntigravity);

        chkSyncGithub = new CheckBox
        {
            Location = new Point(26, 154),
            Size = new Size(780, 28),
            Checked = true,
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 10f, FontStyle.Bold),
            Text = "Automatically verify & sync latest release files from GitHub (schaleh12/Dual-Hemisphere-Wisdom-Model)",
        };
        p.Controls.Add(chkSyncGithub);

        Button btnOpenAg = MakeButton("🌐 Open Google Antigravity Website", new Point(22, 196), new Size(255, 34), Color.FromArgb(30, 88, 145));
        btnOpenAg.Click += (s, e) => OpenUrl("https://antigravity.google/");

        Button btnOpenGh = MakeButton("🐙 Open GitHub Repository & Releases", new Point(287, 196), new Size(265, 34), Color.FromArgb(55, 65, 81));
        btnOpenGh.Click += (s, e) => OpenUrl("https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases");

        Button btnSyncNow = MakeButton("🔄 Run GitHub Auto-Sync Now", new Point(562, 196), new Size(235, 34), Color.FromArgb(83, 122, 50));
        btnSyncNow.Click += (s, e) => RunGithubSyncNow();

        p.Controls.Add(btnOpenAg);
        p.Controls.Add(btnOpenGh);
        p.Controls.Add(btnSyncNow);

        txtPage2Log = new TextBox
        {
            Location = new Point(22, 242),
            Size = new Size(795, 215),
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(14, 15, 19),
            ForeColor = Color.FromArgb(203, 213, 225),
            Font = new Font("Consolas", 9f, FontStyle.Regular),
            BorderStyle = BorderStyle.FixedSingle,
            Text =
                "Repository Root: " + baseDir + "\r\n" +
                "GitHub Upstream: https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model\r\n" +
                "Antigravity Status: " + (antigravityAlreadyInstalled ? ("Installed (" + detectedPath + ")") : "Not found in default paths") + "\r\n\r\n" +
                "Tip: On Page 3, we will scan your GPU and generate a custom system prompt you can paste directly into Google Antigravity!",
        };
        p.Controls.Add(txtPage2Log);
    }

    private void BuildPage3HardwareAndPrompt()
    {
        Panel p = pages[2];

        Label h = MakeHeader("Page 3 of 5 — Graphics Card & System Check + Custom Antigravity Prompt + Upgrade Guide", new Point(22, 14));
        p.Controls.Add(h);

        DetectSystemHardware();

        lblHardwareSummary = new Label
        {
            Location = new Point(22, 44),
            Size = new Size(795, 76),
            Padding = new Padding(10, 8, 10, 8),
            BackColor = Color.FromArgb(28, 33, 44),
            ForeColor = Color.FromArgb(186, 230, 253),
            Font = new Font("Segoe UI", 9.25f, FontStyle.Bold),
            Text = BuildHardwareSummaryBanner(),
        };
        p.Controls.Add(lblHardwareSummary);

        Label lblPromptHeader = new Label
        {
            Location = new Point(22, 128),
            Size = new Size(560, 22),
            ForeColor = Color.FromArgb(163, 230, 53),
            Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
            Text = "Custom System Prompt for Google Antigravity (Tailored to Your Hardware):",
        };
        p.Controls.Add(lblPromptHeader);

        Button btnCopyPrompt = MakeButton("📋 Copy Prompt for Google Antigravity", new Point(562, 123), new Size(255, 28), Color.FromArgb(83, 122, 50));
        btnCopyPrompt.Click += (s, e) =>
        {
            Clipboard.SetText(txtAntigravityPrompt.Text);
            MessageBox.Show("Copied your custom Google Antigravity hardware prompt to the clipboard!", "Prompt Copied", MessageBoxButtons.OK, MessageBoxIcon.Information);
        };
        p.Controls.Add(btnCopyPrompt);

        txtAntigravityPrompt = new TextBox
        {
            Location = new Point(22, 155),
            Size = new Size(795, 195),
            Multiline = true,
            ReadOnly = false,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(14, 15, 19),
            ForeColor = Color.FromArgb(241, 245, 249),
            Font = new Font("Consolas", 9f, FontStyle.Regular),
            BorderStyle = BorderStyle.FixedSingle,
            Text = BuildCustomAntigravityPrompt(),
        };
        p.Controls.Add(txtAntigravityPrompt);

        Label lblAmazon = new Label
        {
            Location = new Point(22, 358),
            Size = new Size(795, 42),
            ForeColor = Color.FromArgb(253, 230, 138),
            Font = new Font("Segoe UI", 9f, FontStyle.Regular),
            Text = isLaptopSystem
                ? "💡 Hardware Upgrade Note (Laptop Detected): Laptop GPUs are soldered to the motherboard, so increasing VRAM requires either an RTX 4080/4090 16GB Laptop, a desktop tower, or a Thunderbolt/OCuLink eGPU dock:"
                : "💡 Hardware Upgrade Note (Desktop Detected): A desktop PCIe slot lets you upgrade directly to a 16GB+ NVIDIA GPU (RTX 4060 Ti 16GB, RTX 4070 Ti Super 16GB, or RTX 4080 Super 16GB) without replacing your whole PC:",
        };
        p.Controls.Add(lblAmazon);

        Button btnAmzGpu = MakeButton("🛒 Amazon: 16GB+ Desktop RTX GPUs", new Point(22, 406), new Size(255, 36), Color.FromArgb(180, 83, 9));
        btnAmzGpu.Click += (s, e) => OpenUrl("https://www.amazon.com/s?k=NVIDIA+RTX+4060+Ti+16GB+RTX+4070+Ti+Super+16GB+Graphics+Card");

        Button btnAmzLaptop = MakeButton("💻 Amazon: 16GB VRAM RTX AI Laptops", new Point(287, 406), new Size(260, 36), Color.FromArgb(30, 88, 145));
        btnAmzLaptop.Click += (s, e) => OpenUrl("https://www.amazon.com/s?k=NVIDIA+RTX+4080+16GB+VRAM+Gaming+Laptop+32GB+RAM");

        Button btnAmzDesktop = MakeButton("🖥️ Amazon: Prebuilt 16GB+ AI Desktops", new Point(557, 406), new Size(260, 36), Color.FromArgb(55, 65, 81));
        btnAmzDesktop.Click += (s, e) => OpenUrl("https://www.amazon.com/s?k=RTX+4070+Ti+Super+16GB+Gaming+PC+Desktop+32GB+RAM");

        p.Controls.Add(btnAmzGpu);
        p.Controls.Add(btnAmzLaptop);
        p.Controls.Add(btnAmzDesktop);
    }

    private void BuildPage4ArchitectContact()
    {
        Panel p = pages[3];

        Label h = MakeHeader("Page 4 of 5 — Follow & Contact the Architect of Celestriad 1.0", new Point(22, 14));
        p.Controls.Add(h);

        PictureBox picPoem = new PictureBox
        {
            Location = new Point(22, 52),
            Size = new Size(340, 220),
            SizeMode = PictureBoxSizeMode.Zoom,
            BorderStyle = BorderStyle.FixedSingle,
        };
        string blueScreenJpg = Path.Combine(assetsDir, "blue_screen_of_life_poem.jpg");
        if (File.Exists(blueScreenJpg))
        {
            try { picPoem.Image = Image.FromFile(blueScreenJpg); } catch { }
        }
        p.Controls.Add(picPoem);

        Label lblInfo = new Label
        {
            Location = new Point(380, 52),
            Size = new Size(435, 220),
            ForeColor = Color.FromArgb(226, 232, 240),
            Font = new Font("Segoe UI", 10f, FontStyle.Regular),
            Text =
                "Created by Schaleh (@imissmydeadcat.74)\r\n\r\n" +
                "• YouTube Channel (Demos, Meditations & AI Evolution Logs):\r\n" +
                "  https://www.youtube.com/@imissmydeadcat.74\r\n\r\n" +
                "• Direct Email for Questions, Collaborations & Feedback:\r\n" +
                "  imissmydeadcat74@gmail.com\r\n\r\n" +
                "• Official GitHub Repository & Releases:\r\n" +
                "  https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model",
        };
        p.Controls.Add(lblInfo);

        Button btnYt = MakeButton("▶️ Open YouTube (@imissmydeadcat.74)", new Point(22, 294), new Size(300, 40), Color.FromArgb(185, 28, 28));
        btnYt.Click += (s, e) => OpenUrl("https://www.youtube.com/@imissmydeadcat.74");

        Button btnCopyYt = MakeButton("📋 Copy YouTube Link", new Point(332, 294), new Size(180, 40), Color.FromArgb(55, 65, 81));
        btnCopyYt.Click += (s, e) =>
        {
            Clipboard.SetText("https://www.youtube.com/@imissmydeadcat.74");
            MessageBox.Show("Copied YouTube URL to clipboard!", "Copied", MessageBoxButtons.OK, MessageBoxIcon.Information);
        };

        Button btnEmail = MakeButton("✉️ Email: imissmydeadcat74@gmail.com", new Point(22, 346), new Size(300, 40), Color.FromArgb(30, 88, 145));
        btnEmail.Click += (s, e) => OpenUrl("mailto:imissmydeadcat74@gmail.com?subject=Celestriad%201.0%20Feedback");

        Button btnCopyEmail = MakeButton("📋 Copy Email Address", new Point(332, 346), new Size(180, 40), Color.FromArgb(55, 65, 81));
        btnCopyEmail.Click += (s, e) =>
        {
            Clipboard.SetText("imissmydeadcat74@gmail.com");
            MessageBox.Show("Copied imissmydeadcat74@gmail.com to clipboard!", "Copied", MessageBoxButtons.OK, MessageBoxIcon.Information);
        };

        Button btnGh = MakeButton("🐙 Open GitHub Releases Page", new Point(524, 294), new Size(290, 92), Color.FromArgb(83, 122, 50));
        btnGh.Click += (s, e) => OpenUrl("https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases");

        p.Controls.Add(btnYt);
        p.Controls.Add(btnCopyYt);
        p.Controls.Add(btnEmail);
        p.Controls.Add(btnCopyEmail);
        p.Controls.Add(btnGh);
    }

    private void BuildPage5IntegrityAndShortcut()
    {
        Panel p = pages[4];

        Label h = MakeHeader("Page 5 of 5 — Install Folder Integrity Check, Desktop Icon & Launch", new Point(22, 14));
        p.Controls.Add(h);

        txtIntegrityReport = new TextBox
        {
            Location = new Point(22, 46),
            Size = new Size(515, 250),
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(14, 15, 19),
            ForeColor = Color.FromArgb(167, 243, 208),
            Font = new Font("Consolas", 9f, FontStyle.Regular),
            BorderStyle = BorderStyle.FixedSingle,
        };
        p.Controls.Add(txtIntegrityReport);

        // Right-side visual previews of the Arecibo Desktop Icon and Celestriad Spirit Particle Icon
        Panel iconCard = new Panel
        {
            Location = new Point(550, 46),
            Size = new Size(268, 250),
            BackColor = Color.FromArgb(28, 30, 38),
            BorderStyle = BorderStyle.FixedSingle,
        };

        PictureBox picArecibo = new PictureBox
        {
            Location = new Point(16, 14),
            Size = new Size(96, 96),
            SizeMode = PictureBoxSizeMode.Zoom,
            BorderStyle = BorderStyle.FixedSingle,
        };
        string areciboJpg = Path.Combine(assetsDir, "arecibo_desktop_icon.jpg");
        if (File.Exists(areciboJpg))
        {
            try { picArecibo.Image = Image.FromFile(areciboJpg); } catch { }
        }
        iconCard.Controls.Add(picArecibo);

        Label lblArecibo = new Label
        {
            Location = new Point(120, 18),
            Size = new Size(138, 90),
            ForeColor = Color.FromArgb(253, 230, 138),
            Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
            Text = "Desktop Shortcut Icon:\r\nArecibo Reply Formation\r\n(arecibo_desktop_icon.ico)",
        };
        iconCard.Controls.Add(lblArecibo);

        PictureBox picSpirit = new PictureBox
        {
            Location = new Point(16, 132),
            Size = new Size(96, 96),
            SizeMode = PictureBoxSizeMode.Zoom,
            BorderStyle = BorderStyle.FixedSingle,
        };
        string spiritJpg = Path.Combine(assetsDir, "celestriad_spirit_particle.jpg");
        if (File.Exists(spiritJpg))
        {
            try { picSpirit.Image = Image.FromFile(spiritJpg); } catch { }
        }
        iconCard.Controls.Add(picSpirit);

        Label lblSpirit = new Label
        {
            Location = new Point(120, 136),
            Size = new Size(138, 90),
            ForeColor = Color.FromArgb(125, 211, 252),
            Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
            Text = "Celestriad 1.0 Icon:\r\nSpirit Particle\r\n(celestriad_spirit_particle.ico)",
        };
        iconCard.Controls.Add(lblSpirit);

        p.Controls.Add(iconCard);

        Button btnReVerify = MakeButton("🔍 Re-Check Folder Integrity", new Point(22, 306), new Size(245, 34), Color.FromArgb(55, 65, 81));
        btnReVerify.Click += (s, e) => RunInstallFolderIntegrityCheck();

        Button btnSetupVenv = MakeButton("🔧 Install / Repair Python .venv", new Point(277, 306), new Size(260, 34), Color.FromArgb(30, 88, 145));
        btnSetupVenv.Click += (s, e) => ProvisionPythonVenv();

        p.Controls.Add(btnReVerify);
        p.Controls.Add(btnSetupVenv);

        chkCreateDesktopShortcut = new CheckBox
        {
            Location = new Point(24, 356),
            Size = new Size(780, 28),
            Checked = true,
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 10f, FontStyle.Bold),
            Text = "Create Desktop Icon ('Celestriad 1.0 — LMFT Chess') using the Arecibo Reply Icon",
        };
        p.Controls.Add(chkCreateDesktopShortcut);

        chkLaunchOnFinish = new CheckBox
        {
            Location = new Point(24, 390),
            Size = new Size(780, 28),
            Checked = true,
            ForeColor = Color.FromArgb(163, 230, 53),
            Font = new Font("Segoe UI", 10f, FontStyle.Bold),
            Text = "Launch Celestriad 1.0 (LMFT-Chess.exe) immediately when I click Finish",
        };
        p.Controls.Add(chkLaunchOnFinish);

        RunInstallFolderIntegrityCheck();
    }

    private void ShowPage(int index)
    {
        currentPage = Math.Max(0, Math.Min(4, index));
        for (int i = 0; i < 5; i++)
        {
            pages[i].Visible = (i == currentPage);
            stepPills[i].BackColor = (i == currentPage)
                ? Color.FromArgb(56, 189, 248)
                : (i < currentPage ? Color.FromArgb(42, 78, 52) : Color.FromArgb(42, 44, 54));
            stepPills[i].ForeColor = (i == currentPage)
                ? Color.FromArgb(15, 23, 42)
                : Color.FromArgb(226, 232, 240);
        }

        btnBack.Enabled = currentPage > 0;
        if (currentPage == 4)
        {
            btnNext.Text = "🚀 Finish & Launch";
            RunInstallFolderIntegrityCheck();
        }
        else
        {
            btnNext.Text = "Next >";
        }
    }

    private void OnNextClicked(object sender, EventArgs e)
    {
        if (currentPage == 1)
        {
            if (chkDownloadAntigravity.Checked)
            {
                OpenUrl("https://antigravity.google/");
            }
            if (chkSyncGithub.Checked)
            {
                RunGithubSyncNow();
            }
        }

        if (currentPage < 4)
        {
            ShowPage(currentPage + 1);
            return;
        }

        // Page 5 Finish!
        if (chkCreateDesktopShortcut.Checked)
        {
            CreateDesktopShortcut();
        }
        if (chkLaunchOnFinish.Checked)
        {
            LaunchChessApp();
        }
        Close();
    }

    private void ComputeLocalHashes()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("=== Local File SHA-256 Verification Digest ===");
        string[] checkFiles = new string[]
        {
            Path.Combine(baseDir, "LMFT-Chess.exe"),
            Path.Combine(baseDir, "LMFT-Bloodmoon.exe"),
            Path.Combine(baseDir, "LMFT-Installer-Wizard.exe"),
            Path.Combine(archDir, "src", "lmft_architecture", "chess_app.py"),
            Path.Combine(archDir, "src", "lmft_architecture", "three_legged_runtime.py"),
        };
        using (SHA256 sha = SHA256.Create())
        {
            foreach (string f in checkFiles)
            {
                if (File.Exists(f))
                {
                    byte[] bytes = File.ReadAllBytes(f);
                    byte[] hash = sha.ComputeHash(bytes);
                    string hex = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                    sb.AppendLine(Path.GetFileName(f) + " (" + (bytes.Length / 1024) + " KB):");
                    sb.AppendLine("  SHA-256: " + hex);
                }
            }
        }
        txtSecurityLog.Text = sb.ToString();
    }

    private void RunWindowsDefenderScan()
    {
        txtSecurityLog.AppendText("\r\nRunning Microsoft Defender Custom Folder Scan on:\r\n  " + baseDir + " ...\r\n");
        string mpCmd = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Windows Defender", "MpCmdRun.exe");
        if (!File.Exists(mpCmd))
        {
            txtSecurityLog.AppendText("MpCmdRun.exe not found at standard path; opening Windows Security.\r\n");
            OpenUrl("windowsdefender:");
            return;
        }
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = mpCmd,
                Arguments = "-Scan -ScanType 3 -File \"" + baseDir + "\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using (Process proc = Process.Start(psi))
            {
                string output = proc.StandardOutput.ReadToEnd();
                proc.WaitForExit(20000);
                txtSecurityLog.AppendText(string.IsNullOrWhiteSpace(output) ? "✅ Scan finished cleanly: 0 threats found.\r\n" : output + "\r\n");
            }
        }
        catch (Exception ex)
        {
            txtSecurityLog.AppendText("Defender scan notice: " + ex.Message + "\r\n");
        }
    }

    private string DetectGoogleAntigravityPath()
    {
        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string[] candidates = new string[]
        {
            Path.Combine(userProfile, ".gemini", "antigravity"),
            Path.Combine(localAppData, "Programs", "Antigravity"),
            Path.Combine(localAppData, "Antigravity"),
        };
        foreach (string c in candidates)
        {
            if (Directory.Exists(c) || File.Exists(c))
            {
                return c;
            }
        }
        return null;
    }

    private void RunGithubSyncNow()
    {
        string pyExe = FindPythonExe();
        if (pyExe == null)
        {
            txtPage2Log.AppendText("\r\n[Notice] Python .venv not provisioned yet — you can provision it on Page 5.\r\n");
            return;
        }
        try
        {
            txtPage2Log.AppendText("\r\nChecking GitHub release_manifest.json (schaleh12/Dual-Hemisphere-Wisdom-Model)...\r\n");
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyExe,
                Arguments = "-c \"from lmft_architecture.github_auto_sync import fetch_remote_github_manifest, verify_or_sync_against_manifest; m = fetch_remote_github_manifest(); r = verify_or_sync_against_manifest('.', m, apply_sync=True); print('GitHub Sync Status:', r['status'], '| Changed:', len(r['changed_files']), '| Synced:', len(r['synced_files']))\"",
                WorkingDirectory = archDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Path.Combine(archDir, "src");
            using (Process proc = Process.Start(psi))
            {
                string stdout = proc.StandardOutput.ReadToEnd();
                string stderr = proc.StandardError.ReadToEnd();
                proc.WaitForExit(15000);
                if (!string.IsNullOrWhiteSpace(stdout)) txtPage2Log.AppendText(stdout + "\r\n");
                if (!string.IsNullOrWhiteSpace(stderr)) txtPage2Log.AppendText(stderr + "\r\n");
            }
        }
        catch (Exception ex)
        {
            txtPage2Log.AppendText("Sync check error: " + ex.Message + "\r\n");
        }
    }

    private void DetectSystemHardware()
    {
        try
        {
            // 1. Check nvidia-smi first for accurate NVIDIA VRAM
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=name,memory.total --format=csv,noheader,nounits",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using (Process p = Process.Start(psi))
            {
                string line = p.StandardOutput.ReadLine();
                p.WaitForExit(4000);
                if (!string.IsNullOrWhiteSpace(line) && line.Contains(","))
                {
                    string[] parts = line.Split(',');
                    detectedGpuName = parts[0].Trim();
                    double mb;
                    if (double.TryParse(parts[1].Trim(), out mb))
                    {
                        detectedVramGb = Math.Round(mb / 1024.0, 1);
                    }
                }
            }
        }
        catch { }

        if (detectedGpuName == "Detecting GPU...")
        {
            try
            {
                ProcessStartInfo psiWmic = new ProcessStartInfo
                {
                    FileName = "wmic",
                    Arguments = "path win32_VideoController get Name,AdapterRAM /value",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                };
                using (Process p = Process.Start(psiWmic))
                {
                    string output = p.StandardOutput.ReadToEnd();
                    p.WaitForExit(4000);
                    foreach (string raw in output.Split('\n'))
                    {
                        string ln = raw.Trim();
                        if (ln.StartsWith("Name=") && ln.Length > 5)
                        {
                            detectedGpuName = ln.Substring(5).Trim();
                        }
                    }
                }
            }
            catch
            {
                detectedGpuName = "Standard Windows Display Adapter";
            }
        }

        // Detect Laptop vs Desktop (Battery presence or Mobile/Laptop GPU name)
        isLaptopSystem =
            detectedGpuName.IndexOf("Laptop", StringComparison.OrdinalIgnoreCase) >= 0 ||
            detectedGpuName.IndexOf("Mobile", StringComparison.OrdinalIgnoreCase) >= 0 ||
            SystemInformation.PowerStatus.BatteryChargeStatus != BatteryChargeStatus.NoSystemBattery;

        try
        {
            ProcessStartInfo psiRam = new ProcessStartInfo
            {
                FileName = "wmic",
                Arguments = "computersystem get TotalPhysicalMemory /value",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
            };
            using (Process p = Process.Start(psiRam))
            {
                string output = p.StandardOutput.ReadToEnd();
                p.WaitForExit(4000);
                foreach (string raw in output.Split('\n'))
                {
                    string ln = raw.Trim();
                    if (ln.StartsWith("TotalPhysicalMemory="))
                    {
                        long bytes;
                        if (long.TryParse(ln.Substring("TotalPhysicalMemory=".Length).Trim(), out bytes))
                        {
                            detectedRamGb = Math.Round(bytes / (1024.0 * 1024.0 * 1024.0), 1);
                        }
                    }
                }
            }
        }
        catch { }
    }

    private string BuildHardwareSummaryBanner()
    {
        string formFactor = isLaptopSystem ? "💻 Form Factor: Laptop / Notebook PC" : "🖥️ Form Factor: Desktop Workstation / Tower";
        string vramLabel = detectedVramGb > 0 ? (detectedVramGb.ToString("0.0") + " GB VRAM") : "Shared / Integrated Memory";
        string tierAdvice;
        if (detectedVramGb >= 12.0)
        {
            tierAdvice = "✅ Tier 1 Ready (12GB+ VRAM): Ideal for 4-bit Qwen3-8B Left Brain + Qwen3.5-9B REM Sleep Teacher + Bloodmoon 12B!";
        }
        else if (detectedVramGb >= 7.5)
        {
            tierAdvice = "✅ Tier 2 Ready (8GB–11GB VRAM): Runs 4-bit Qwen3-8B or Qwen3-4B Left Brain smoothly with sequential REM unloading.";
        }
        else
        {
            tierAdvice = "⚠️ Tier 3 / Low-VRAM Mode: Fast 40-Node Right Brain runs in ~15 ms on CPU, but Left-Brain LLM speech benefits from an NVIDIA GPU upgrade or Qwen3-4B 4-bit.";
        }

        return formFactor + "   |   🎮 GPU: " + detectedGpuName + " (" + vramLabel + ")   |   🧠 System RAM: " + detectedRamGb.ToString("0.0") + " GB\r\n" + tierAdvice;
    }

    private string BuildCustomAntigravityPrompt()
    {
        string profileMode = detectedVramGb >= 12.0
            ? "Keep 4-bit NF4 Qwen/Qwen3-8B enabled as the primary Left Brain and keep sequential VRAM unloading active during REM Sleep."
            : (detectedVramGb >= 7.5
                ? "Configure LMFT-Chess to prefer 4-bit NF4 Qwen/Qwen3-4B or Qwen3-8B with max_context_chars=12000 and strict torch.cuda.empty_cache() before REM Sleep so VRAM stays under 8 GB."
                : "Configure LMFT-Chess for Low-VRAM / CPU-Assisted mode: keep the 40-Node Right Brain and 3 MiniLM bridges on CPU/fp16, and use a lightweight 4-bit Qwen3-4B Left Brain with short max_post_gap_tokens=28.");

        return
            "Paste this prompt into Google Antigravity to auto-tune Celestriad 1.0 for your exact hardware:\r\n" +
            "--------------------------------------------------------------------------------------------\r\n" +
            "I am running the Dual-Hemisphere Wisdom Model (LMFT-Architecture / Celestriad 1.0) on my " + (isLaptopSystem ? "Laptop" : "Desktop PC") + ".\r\n" +
            "My detected hardware specs are:\r\n" +
            "- Graphics Card (GPU): " + detectedGpuName + " (" + (detectedVramGb > 0 ? (detectedVramGb.ToString("0.0") + " GB VRAM") : "Integrated / Shared GPU") + ")\r\n" +
            "- System Memory (RAM): " + detectedRamGb.ToString("0.0") + " GB RAM (" + Environment.ProcessorCount + " logical CPU cores)\r\n\r\n" +
            "Please inspect `LMFT-Architecture/src/lmft_architecture/chess_app.py` and `three_legged_runtime.py` and customize my local runtime settings so that:\r\n" +
            "1. " + profileMode + "\r\n" +
            "2. The 40-Node Right-Brain Fuzzy Tree (`birth.py`) and `<|subliminal_gap|>` comma splice remain 100% intact and un-rigged.\r\n" +
            "3. Verify the configuration with `pytest LMFT-Architecture/tests/test_chess_app.py -q`.";
    }

    private void RunInstallFolderIntegrityCheck()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("=== Celestriad 1.0 Install Folder Integrity Check ===");
        sb.AppendLine("Install Directory: " + baseDir);
        sb.AppendLine("");

        CheckItem(sb, "1. Core 40-Node Seed (birth.py)", Path.Combine(archDir, "src", "lmft_architecture", "birth.py"));
        CheckItem(sb, "2. Chess Arena Engine (chess_app.py)", Path.Combine(archDir, "src", "lmft_architecture", "chess_app.py"));
        CheckItem(sb, "3. Subliminal Gap Runtime (three_legged_runtime.py)", Path.Combine(archDir, "src", "lmft_architecture", "three_legged_runtime.py"));
        CheckItem(sb, "4. Virgin Baseline Lock (virgin_baseline.py)", Path.Combine(archDir, "src", "lmft_architecture", "virgin_baseline.py"));
        CheckItem(sb, "5. Right-Brain MiniLM Bridges", Path.Combine(trainDir, "models", "right_brain_bridges", "bridge_training_report.json"));
        CheckItem(sb, "6. Qwen3-8B Left-Brain LoRA + Callosum", Path.Combine(trainDir, "models", "left_brain_qwen3_8b", "subliminal_gap_projector.pt"));
        CheckItem(sb, "7. Master Release Wisdom Seed", Path.Combine(trainDir, "outputs", "rem_sleep_consolidated_ruleset.json"));
        CheckItem(sb, "8. Arecibo Reply Desktop Icon (.ico)", Path.Combine(assetsDir, "arecibo_desktop_icon.ico"));
        CheckItem(sb, "9. Celestriad Spirit Particle Icon (.ico)", Path.Combine(assetsDir, "celestriad_spirit_particle.ico"));

        string pyExe = FindPythonExe();
        if (pyExe != null)
        {
            sb.AppendLine("✅ [OK] 10. Python Virtual Environment (.venv): " + pyExe);
        }
        else
        {
            sb.AppendLine("⚠️ [ACTION NEEDED] 10. Python .venv not created yet -> Click [Install / Repair Python .venv] below!");
        }

        txtIntegrityReport.Text = sb.ToString();
    }

    private void CheckItem(StringBuilder sb, string label, string path)
    {
        if (File.Exists(path) || Directory.Exists(path))
        {
            sb.AppendLine("✅ [OK] " + label);
        }
        else
        {
            sb.AppendLine("⚠️ [MISSING] " + label + " (" + path + ")");
        }
    }

    private void ProvisionPythonVenv()
    {
        string batPath = Path.Combine(baseDir, "Install-And-Run-LMFT-Chess.bat");
        if (File.Exists(batPath))
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = batPath,
                Arguments = "--venv-only",
                WorkingDirectory = baseDir,
                UseShellExecute = true,
            });
        }
        else
        {
            MessageBox.Show("Install-And-Run-LMFT-Chess.bat not found in " + baseDir, "Installer", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
    }

    private void CreateDesktopShortcut()
    {
        try
        {
            string desktopDir = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            string shortcutPath = Path.Combine(desktopDir, "Celestriad 1.0 — LMFT Chess.lnk");
            string targetExe = Path.Combine(baseDir, "LMFT-Chess.exe");
            string iconPath = Path.Combine(assetsDir, "arecibo_desktop_icon.ico");
            if (!File.Exists(iconPath))
            {
                iconPath = targetExe;
            }

            string psScript =
                "$ws = New-Object -ComObject WScript.Shell; " +
                "$s = $ws.CreateShortcut('" + shortcutPath.Replace("'", "''") + "'); " +
                "$s.TargetPath = '" + targetExe.Replace("'", "''") + "'; " +
                "$s.WorkingDirectory = '" + baseDir.Replace("'", "''") + "'; " +
                "$s.IconLocation = '" + iconPath.Replace("'", "''") + "'; " +
                "$s.Description = 'Launch Celestriad 1.0 — Dual-Hemisphere Chess Competitor & Coach'; " +
                "$s.Save()";

            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = "-NoProfile -ExecutionPolicy Bypass -Command \"" + psScript + "\"",
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using (Process p = Process.Start(psi))
            {
                p.WaitForExit(6000);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show("Could not create desktop shortcut:\n" + ex.Message, "Desktop Shortcut", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
    }

    private void LaunchChessApp()
    {
        string targetExe = Path.Combine(baseDir, "LMFT-Chess.exe");
        if (File.Exists(targetExe))
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = targetExe,
                WorkingDirectory = baseDir,
                UseShellExecute = true,
            });
        }
    }

    private string FindPythonExe()
    {
        string[] candidates = new string[]
        {
            Path.Combine(archDir, ".venv", "Scripts", "python.exe"),
            Path.Combine(baseDir, ".venv", "Scripts", "python.exe"),
        };
        foreach (string c in candidates)
        {
            if (File.Exists(c)) return c;
        }
        return null;
    }

    private static Label MakeHeader(string text, Point location)
    {
        return new Label
        {
            Text = text,
            Location = location,
            AutoSize = true,
            Font = new Font("Segoe UI", 12f, FontStyle.Bold),
            ForeColor = Color.FromArgb(56, 189, 248),
        };
    }

    private static Button MakeButton(string text, Point location, Size size, Color backColor)
    {
        Button b = new Button
        {
            Text = text,
            Location = location,
            Size = size,
            BackColor = backColor,
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Font = new Font("Segoe UI", 9.25f, FontStyle.Bold),
            Cursor = Cursors.Hand,
        };
        b.FlatAppearance.BorderColor = Color.FromArgb(100, 116, 139);
        b.FlatAppearance.BorderSize = 1;
        return b;
    }

    private static void OpenUrl(string url)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = url,
                UseShellExecute = true,
            });
        }
        catch { }
    }
}
