using UnityEngine;

public class FeedbackTester : MonoBehaviour
{
    void Start()
    {
        // Use the new API introduced in Unity 2023+
        var f = Object.FindFirstObjectByType<FeedbackUI>();
        if (f != null)
            f.Enqueue("✅ Feedback UI test successful!");
        else
            Debug.LogWarning("No FeedbackUI found in scene!");
    }
}
