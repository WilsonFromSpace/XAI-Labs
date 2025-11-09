using UnityEngine;

public class TogglePanel : MonoBehaviour
{
    public GameObject targetPanel;

    public void Toggle()
    {
        if (targetPanel != null)
            targetPanel.SetActive(!targetPanel.activeSelf);
    }
}
