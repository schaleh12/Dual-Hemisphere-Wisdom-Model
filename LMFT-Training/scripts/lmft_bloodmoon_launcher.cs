using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string archDir = Directory.Exists(Path.Combine(baseDir, "LMFT-Architecture"))
                ? Path.Combine(baseDir, "LMFT-Architecture")
                : baseDir;

            string[] pythonCandidates = new string[]
            {
                Path.Combine(archDir, ".venv", "Scripts", "python.exe"),
                Path.Combine(archDir, ".venv", "Scripts", "pythonw.exe"),
                Path.Combine(baseDir, ".venv", "Scripts", "python.exe"),
                Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe"),
            };

            string exePath = "python.exe";
            foreach (string cand in pythonCandidates)
            {
                if (File.Exists(cand))
                {
                    exePath = cand;
                    break;
                }
            }

            string extraArgs = args.Length > 0 ? " " + string.Join(" ", args) : "";
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = "-m lmft_architecture.bloodmoon_app --single-window" + extraArgs,
                WorkingDirectory = archDir,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Path.Combine(archDir, "src");
            psi.EnvironmentVariables["HF_HUB_DISABLE_PROGRESS_BARS"] = "1";
            psi.EnvironmentVariables["TQDM_DISABLE"] = "1";
            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Could not launch LMFT-Bloodmoon:\n" + ex.Message,
                "LMFT-Bloodmoon Launcher",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }
}
