using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class EvalLogger : MonoBehaviour
{
    public static EvalLogger Instance { get; private set; }

    // Backwards-compat alias for old calls: EvalLogger.I
    public static EvalLogger I => Instance;

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

    /// <summary>
    /// High-level info / lifecycle events.
    /// Example: EvalLogger.Instance.Info("TutorialShown");
    /// </summary>
    public void Info(string evt, Dictionary<string, object> payload = null)
    {
        Log("Info", evt, payload);
    }

    /// <summary>
    /// User actions: button clicks, slider changes, mission attempts.
    /// Example: EvalLogger.Instance.ActionEvent("S1_AdjustWeight", payload);
    /// </summary>
    public void ActionEvent(string evt, Dictionary<string, object> payload = null)
    {
        Log("Action", evt, payload);
    }

    /// <summary>
    /// Metrics: F-score per run, time taken, attempts, etc.
    /// Example: EvalLogger.Instance.Metric("S1_MissionCompleted", payload);
    /// </summary>
    public void Metric(string evt, Dictionary<string, object> payload = null)
    {
        Log("Metric", evt, payload);
    }

    private void Log(string eventType, string key, Dictionary<string, object> payload)
    {
        if (EventLogger.Instance == null)
            return; // do not hard-crash if logger missing

        string extra = "";
        float? fScore = null;

        if (payload != null && payload.Count > 0)
        {
            // pull fScore into dedicated column if present
            if (payload.TryGetValue("fScore", out var fVal))
            {
                if (float.TryParse(fVal.ToString(), out var parsed))
                    fScore = parsed;
            }

            // everything else into extra="k1=v1;k2=v2"
            extra = string.Join(";", payload
                .Where(kv => kv.Key != "fScore")
                .Select(kv => kv.Key + "=" + kv.Value));
        }

        EventLogger.Instance.LogEvent(
            eventType: eventType,
            key: key,
            value: "",
            fScore: fScore,
            extra: extra
        );
    }
}
