using UnityEngine;
using UnityEngine.UI;
using TMPro;

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

        // Wire activation dropdown (TryVariantsCount in other scenes, still useful here)
        if (activationDropdown)
            activationDropdown.onValueChanged.AddListener(OnActivationChanged);

        // Wire step button so each press counts as a weight_adjust
        if (stepButton)
            stepButton.onClick.AddListener(() => ReportWeightAdjust());

        // Wire weight sliders so each change counts as a weight_adjust
        if (weightSliders != null)
        {
            foreach (var s in weightSliders)
            {
                if (!s) continue;
                s.onValueChanged.AddListener(_ => ReportWeightAdjust());
            }
        }

        // Optional: if you have a dedicated Win/Complete button in the UI
        if (winButton)
            winButton.onClick.AddListener(ReportSceneFinished);
    }

    // --- Public helpers you can call from anywhere (e.g., UnityEvents) ---

    public void ReportFaithfulness(float F01)
    {
        tracker?.ReportFaithfulness(Mathf.Clamp01(F01));
    }

    public void ReportSceneFinished()
    {
        tracker?.ReportSceneFinish();
    }

    public void ReportWeightAdjust()
    {
        tracker?.ReportAction("weight_adjust");
    }

    public void OnActivationChanged(int index)
    {
        // If your activation enum matches 0=Tanh,1=ReLU,2=Sigmoid adjust the names as you wish
        string name = index switch
        {
            0 => "Tanh",
            1 => "ReLU",
            2 => "Sigmoid",
            _ => $"Act_{index}"
        };
        tracker?.ReportTriedVariant(name);

        // You can also re-check faithfulness right after activation changes by calling ReportFaithfulness(...)
        // from wherever you compute accuracy/F.
    }
}
