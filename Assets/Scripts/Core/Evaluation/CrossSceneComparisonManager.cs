using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

public class CrossSceneComparisonManager : MonoBehaviour
{
    public static CrossSceneComparisonManager Instance { get; private set; }

    [Serializable]
    public class SceneStats
    {
        public string sceneId;
        public int runs;
        public int successes;
        public float bestF = -1f; // -1 = no F recorded yet
        public float lastF = -1f;

        public float SuccessRate => runs > 0 ? (float)successes / runs : 0f;
    }

    private readonly Dictionary<string, SceneStats> _stats = new Dictionary<string, SceneStats>();

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private SceneStats GetOrCreate(string sceneId)
    {
        if (!_stats.TryGetValue(sceneId, out var s))
        {
            s = new SceneStats { sceneId = sceneId };
            _stats[sceneId] = s;
        }
        return s;
    }

    /// <summary>
    /// Call once per run outcome (success/fail) from a scene.
    /// </summary>
    public void RegisterRun(string sceneId, float? fScore, bool success)
    {
        var s = GetOrCreate(sceneId);

        s.runs++;
        if (success) s.successes++;

        if (fScore.HasValue)
        {
            var F = Mathf.Clamp01(fScore.Value);
            s.lastF = F;
            if (F > s.bestF) s.bestF = F;
        }

        // Mirror into EventLogger for traceability
        EventLogger.Instance?.LogEvent(
            eventType: "RunSummary",
            key: sceneId,
            value: success ? "success" : "fail",
            fScore: fScore,
            extra: $"runs={s.runs};successes={s.successes};bestF={(s.bestF < 0 ? -1f : s.bestF):0.000}"
        );
    }

    public IEnumerable<SceneStats> GetAllStats()
    {
        return _stats.Values.OrderBy(s => s.sceneId);
    }

    /// <summary>
    /// Export aggregated faithfulness + success metrics to CSV for thesis appendix.
    /// File: faithfulness_results.csv in Application.persistentDataPath.
    /// </summary>
    public void SaveToCsv()
    {
        try
        {
            var path = Path.Combine(Application.persistentDataPath, "faithfulness_results.csv");
            var sb = new StringBuilder();

            sb.AppendLine("sceneId;runs;successes;successRate;bestF;lastF");

            foreach (var s in GetAllStats())
            {
                sb.AppendLine(
                    $"{s.sceneId};" +
                    $"{s.runs};" +
                    $"{s.successes};" +
                    $"{s.SuccessRate:0.000};" +
                    $"{(s.bestF < 0 ? -1f : s.bestF):0.000};" +
                    $"{(s.lastF < 0 ? -1f : s.lastF):0.000}"
                );
            }

            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);

            EventLogger.Instance?.LogEvent(
                eventType: "CrossSceneCsvExport",
                key: "faithfulness_results.csv",
                value: path
            );

            Debug.Log("[CrossSceneComparisonManager] Saved CSV to: " + path);
        }
        catch (Exception e)
        {
            Debug.LogWarning("[CrossSceneComparisonManager] Failed to save CSV: " + e.Message);
        }
    }
}
