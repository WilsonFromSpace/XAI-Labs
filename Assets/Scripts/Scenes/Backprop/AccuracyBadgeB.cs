// Assets/Scripts/Scenes/Backprop/AccuracyBadgeB.cs
using UnityEngine;
using TMPro;
public class AccuracyBadgeB : MonoBehaviour
{
    public TMP_Text txt;
    public void UpdateFrom(MLP mlp, float[,] X, float[,] Y, int step)
    {
        if (!txt || mlp == null) return;
        var (_, P) = mlp.Forward(X, Y);
        int n = P.GetLength(0), correct = 0;
        for (int i = 0; i < n; i++) { bool pred = P[i, 0] >= 0.5f; bool lab = Y[i, 0] >= 0.5f; if (pred == lab) correct++; }
        float acc = 100f * correct / Mathf.Max(1, n);
        txt.text = $"Accuracy: {acc:0.#}%    Step: {step}";
    }
}
