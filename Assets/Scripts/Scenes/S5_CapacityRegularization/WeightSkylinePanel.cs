using UnityEngine;
using UnityEngine.UI;

public class WeightSkylinePanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new(0.08f, 0.08f, 0.1f, 1f),
                 cL1 = new(1f, 0.78f, 0.4f, 1f),
                 cL2 = new(0.6f, 0.85f, 1f, 1f),
                 cZero = new(0.25f, 0.25f, 0.3f, 1f);
    Texture2D tex; const int W = 420, H = 140;

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        img.texture = tex;
    }

    public void Redraw(MLP_Capacity mlp)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc; tex.SetPixels32(px);

        var mags = new System.Collections.Generic.List<float>();
        foreach (var L in mlp.Ls)
            for (int i = 0; i < L.W.GetLength(0); i++)
                for (int j = 0; j < L.W.GetLength(1); j++)
                    mags.Add(Mathf.Abs(L.W[i, j]));

        if (mags.Count == 0) { tex.Apply(false); return; }
        float max = 1e-6f; foreach (var v in mags) if (v > max) max = v;

        int n = mags.Count;
        for (int k = 0; k < n; k++)
        {
            int x0 = Mathf.RoundToInt(k * (W - 1f) / n), x1 = Mathf.RoundToInt((k + 1) * (W - 1f) / n);
            int h = Mathf.RoundToInt((mags[k] / max) * (H - 4));
            Color c = Mathf.Approximately(mags[k], 0f) ? cZero : Color.Lerp(cL2, cL1, 0.5f);
            for (int x = x0; x < Mathf.Max(x0, x1); x++) for (int y = 2; y < 2 + h; y++) tex.SetPixel(x, y, c);
        }
        tex.Apply(false);
    }
}
