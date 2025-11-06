using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class FeedbackUI : MonoBehaviour
{
    [SerializeField] CanvasGroup panel;
    [SerializeField] TextMeshProUGUI label;
    [SerializeField] float showSeconds = 2.2f;
    readonly Queue<string> q = new(); bool busy;

    public void Enqueue(string msg) { q.Enqueue(msg); if (!busy) StartCoroutine(Show()); }

    IEnumerator Show()
    {
        busy = true;
        while (q.Count > 0)
        {
            label.text = q.Dequeue();
            yield return Fade(1, .15f);
            yield return new WaitForSeconds(showSeconds);
            yield return Fade(0, .15f);
        }
        busy = false;
    }

    IEnumerator Fade(float to, float d) { float t = 0, a = panel.alpha; while (t < d) { t += Time.deltaTime; panel.alpha = Mathf.Lerp(a, to, t / d); yield return null; } panel.alpha = to; }
}
