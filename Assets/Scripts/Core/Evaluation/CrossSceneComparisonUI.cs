using System.Linq;
using TMPro;
using UnityEngine;

public class CrossSceneComparisonUI : MonoBehaviour
{
    public Transform tableRoot;
    public GameObject rowPrefab;

    private void OnEnable()
    {
        foreach (Transform child in tableRoot)
            Destroy(child.gameObject);

        if (CrossSceneComparisonManager.Instance == null)
            return;

        var stats = CrossSceneComparisonManager.Instance.GetAllStats().ToList();
        if (stats.Count == 0)
            return;

        foreach (var s in stats)
        {
            var row = Instantiate(rowPrefab, tableRoot);
            var texts = row.GetComponentsInChildren<TMP_Text>();

            // assume order: Scene | Runs | SuccessRate | BestF
            texts[0].text = s.sceneId;
            texts[1].text = s.runs.ToString();
            texts[2].text = (s.SuccessRate * 100f).ToString("F0") + " %";
            texts[3].text = s.bestF < 0 ? "-" : s.bestF.ToString("F3");
        }
    }
}
