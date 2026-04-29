
using System;
using System.Diagnostics;
using System.IO;

class Program
{
    static int Main(string[] args)
    {
        try
        {
            // The EXE lives in Kinetic_Lipidomics (repo root).
            // The CMD lives in Kinetic_Lipidomics\Lipidomics_Kinetics_Workflow2\
            string exeDir = AppContext.BaseDirectory;
            string scriptRel = Path.Combine("Lipidomics_Kinetics_Workflow2", "Lipidomics_Kinetics_Workflow2.cmd");
            string scriptPath = Path.GetFullPath(Path.Combine(exeDir, scriptRel));

            if (!File.Exists(scriptPath))
            {
                Console.Error.WriteLine($"ERROR: Script not found at '{scriptPath}'.");
                return 2;
            }

            // Pass-through args to the .cmd
            string argString = "/c \"" + scriptPath + "\"";
            if (args.Length > 0)
                argString += " " + string.Join(" ", Array.ConvertAll(args, EscapeArg));

            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = argString,
                WorkingDirectory = Path.GetDirectoryName(scriptPath) ?? exeDir,
                UseShellExecute = false,
            };

            using var proc = Process.Start(psi);
            if (proc == null)
            {
                Console.Error.WriteLine("ERROR: Failed to start cmd.exe.");
                return 3;
            }
            proc.WaitForExit();
            return proc.ExitCode;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Launcher error: " + ex.Message);
            return 1;
        }
    }

    static string EscapeArg(string a)
    {
        if (string.IsNullOrEmpty(a)) return "\"\"";
        if (a.IndexOfAny(new[]{' ', '\t', '"'}) == -1) return a;
        return "\"" + a.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
}
