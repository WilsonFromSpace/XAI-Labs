using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

public class EventLogger : MonoBehaviour
{
    public static EventLogger Instance { get; private set; }

    [Header("File Settings")]
    [SerializeField] private string fileName = "log.txt";

    private string _filePath;
    private string _sessionId;

    private const string Separator = ";";

    private void Awake()
    {
        // Singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);

        _filePath = Path.Combine(Application.persistentDataPath, fileName);

        // Initialize file with header if it does not exist
        if (!File.Exists(_filePath))
        {
            var header = new StringBuilder();
            header.Append("timestamp").Append(Separator)
                  .Append("sessionId").Append(Separator)
                  .Append("scene").Append(Separator)
                  .Append("eventType").Append(Separator)
                  .Append("key").Append(Separator)
                  .Append("value").Append(Separator)
                  .Append("fScore").Append(Separator)
                  .Append("extra")
                  .AppendLine();

            File.WriteAllText(_filePath, header.ToString(), Encoding.UTF8);
        }

        StartNewSession();
        SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private void OnDestroy()
    {
        if (Instance == this)
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
        }
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        LogEvent(
            eventType: "SceneLoaded",
            key: "mode",
            value: mode.ToString()
        );
    }

    public void StartNewSession()
    {
        _sessionId = DateTime.UtcNow.ToString("yyyyMMddHHmmssfff", CultureInfo.InvariantCulture);
        LogEvent("SessionStart");
    }

    /// <summary>
    /// Core logging API. Call this from any scene.
    /// </summary>
    public void LogEvent(
        string eventType,
        string key = "",
        string value = "",
        float? fScore = null,
        string extra = ""
    )
    {
        try
        {
            var ts = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
            var sceneName = SceneManager.GetActiveScene().name;

            string fScoreStr = fScore.HasValue
                ? fScore.Value.ToString("F4", CultureInfo.InvariantCulture)
                : "";

            string line =
                Sanitize(ts) + Separator +
                Sanitize(_sessionId) + Separator +
                Sanitize(sceneName) + Separator +
                Sanitize(eventType) + Separator +
                Sanitize(key) + Separator +
                Sanitize(value) + Separator +
                Sanitize(fScoreStr) + Separator +
                Sanitize(extra) +
                Environment.NewLine;

            File.AppendAllText(_filePath, line, Encoding.UTF8);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EventLogger] Failed to write log: {e.Message}");
        }
    }

    private string Sanitize(string input)
    {
        if (string.IsNullOrEmpty(input))
            return "";

        // no newlines, no separator hell
        input = input.Replace("\n", " ").Replace("\r", " ");
        input = input.Replace(Separator, ",");
        return input;
    }

    // Optional: quick helper for debugging path
    public string GetLogFilePath() => _filePath;
}
