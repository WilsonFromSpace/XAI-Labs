using UnityEngine;

public class FaithfulnessExportButton : MonoBehaviour
{
    public void Export()
    {
        if (CrossSceneComparisonManager.Instance != null)
        {
            CrossSceneComparisonManager.Instance.SaveToCsv();
        }
        else
        {
            Debug.LogWarning("[FaithfulnessExportButton] No CrossSceneComparisonManager.Instance found.");
        }
    }
}
