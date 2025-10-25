using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class LossChartMulti : MonoBehaviour
{
    public RawImage img;
    public int capacity = 220;
    public float yMin = 0f;
    public float yMax = 2f;

    public Color colSGD = new Color(0.70f, 0.85f, 1f, 1f);
    public Color colMom = new Color(0.75f, 1f, 0.75f, 1f);
    public Color colAdam = new Color(1f, 0.85f, 0.60f, 1f);

    Texture2D tex;
    readonly List<float> sgd = new();
    readonly List<float> mom = new();
    readonly List<float> adam = new();

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(capacity, 80, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        img.texture = tex;
        Clear();
    }

    public void Push(float lSGD, float lMom, float lAdam)
    {
        sgd.Add(lSGD); mom.Add(lMom); adam.Add(lAdam);
        if (sgd.Count > capacity) { sgd.RemoveAt(0); mom.RemoveAt(0); adam.RemoveAt(0); }
        Redraw();
    }

    public void ClearSeries() { sgd.Clear(); mom.Clear(); adam.Clear(); Clear(); }

    void Redraw()
    {
        Clear();
        DrawSeries(sgd, colSGD);
        DrawSeries(mom, colMom);
        DrawSeries(adam, colAdam);
        tex.Apply(false);
    }

    void DrawSeries(List<float> vals, Color c)
    {
        if (vals.Count < 2) return;
        for (int x = 0; x < vals.Count; x++)
        {
            float t = Mathf.InverseLerp(yMin, yMax, vals[x]);
            int y = Mathf.Clamp(Mathf.RoundToInt(t * (tex.height - 1)), 0, tex.height - 1);
            tex.SetPixel(x, y, c);
            if (x > 0)
            {
                float tPrev = Mathf.InverseLerp(yMin, yMax, vals[x - 1]);
                int yPrev = Mathf.Clamp(Mathf.RoundToInt(tPrev * (tex.height - 1)), 0, tex.height - 1);
                int y0 = Mathf.Min(y, yPrev), y1 = Mathf.Max(y, yPrev);
                for (int yy = y0; yy <= y1; yy++) tex.SetPixel(x, yy, c);
            }
        }
    }

    void Clear()
    {
        var cols = tex.GetPixels32();
        var bg = new Color32(24, 24, 24, 255);
        for (int i = 0; i < cols.Length; i++) cols[i] = bg;
        tex.SetPixels32(cols);
        tex.Apply(false);
    }
}
