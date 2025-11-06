using UnityEngine;

public enum ObjectiveType { FaithfulnessAtLeast, AccuracyAtLeast, LossAtMost, PerformActionCount, TryVariantsCount, FinishUnderSec }
public enum ObjectiveStatus { Locked, Active, Completed }

[CreateAssetMenu(menuName = "XAI/Objective")]
public class Objective : ScriptableObject
{
    [Header("ID")]
    public string objectiveId;  // e.g., "S1_F085"
    public string sceneName;    // e.g., "S1_Backprop"
    public string title;

    [Header("Rule")]
    public ObjectiveType type;
    public float threshold = 0.85f;
    public string actionKey; // used for PerformActionCount
    public int actionTarget = 10;

    [Header("UX")]
    [TextArea] public string onActivateStory;   // tiny sci-fi line
    [TextArea] public string onCompleteToast;   // toast message
}
