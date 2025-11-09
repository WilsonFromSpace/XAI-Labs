using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;

public class S1GamificationBridge : MonoBehaviour
{
    [Header("Links (optional but handy)")]
    [SerializeField] ObjectiveTracker tracker;
    [SerializeField] TMP_Dropdown activationDropdown;
    [SerializeField] Button stepButton;
    [SerializeField] Button winButton;           // hook your “Win”/“Complete” button here if you have one

    [Header("Weight sliders to count as actions")]
    [SerializeField] Slider[] weightSliders;     // add any sliders whose changes mean weight_adjust

    void Awake()
    {
        // Auto-find tracker if not assigned
        if (!tracker) tracker = Object.FindFirstObjectByType<ObjectiveTracker>();

        // Wire activation dropdown
        if (activationDropdown)
            activationDropdown.onValueChanged.AddListener(OnActivationChanged);

        // Wire step button so each press counts as a weight_adjust
        if (stepButton)
            stepButton.onClick.AddListener(() => ReportWeightAdjust());

        // Wire weight sliders
        if (weightSliders != null)
        {
            foreach (var s in weightSliders)
            {
                if (!s) continue;
                s.onValueChanged.AddListener(_ => ReportWeightAdjust());
            }
        }

        // Optional: Win/Complete button
        if (winButton)
            winButton.onClick.AddListener(ReportSceneFinished);
    }

    // --- Public helpers you can call from anywhere (e.g., UnityEvents) ---

    public void ReportFaithfulness(float F01)
    {
        tracker?.ReportFaithfulness(Mathf.Clamp01(F01));

        EventLogger.Instance?.LogEvent(
            eventType: "FaithfulnessUpdated",
            fScore: Mathf.Clamp01(F01)
        );
    }

    public void ReportSceneFinished()
    {
        if (!tracker)
            tracker = Object.FindFirstObjectByType<ObjectiveTracker>();

        tracker?.ReportSceneFinish();

        string sceneId = SceneManager.GetActiveScene().name;

        // Log run completion (no hard F binding here; F is tracked separately)
        EventLogger.Instance?.LogEvent(
            eventType: "RunCompleted",
            key: sceneId,
            value: "success"
        );

        // Cross-scene summary entry (F unknown -> null)
        CrossSceneComparisonManager.Instance?.RegisterRun(
            sceneId: sceneId,
            fScore: null,
            success: true
        );
    }

    public void ReportWeightAdjust()
    {
        tracker?.ReportAction("weight_adjust");

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "weight_adjust"
        );
    }

    public void OnActivationChanged(int index)
    {
        string name = index switch
        {
            0 => "Tanh",
            1 => "ReLU",
            2 => "Sigmoid",
            _ => $"Act_{index}"
        };

        tracker?.ReportTriedVariant(name);

        EventLogger.Instance?.LogEvent(
            eventType: "ParamChange",
            key: "activation",
            value: name
        );
    }
}
