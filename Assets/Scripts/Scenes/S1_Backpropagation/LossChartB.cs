using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class LossChartB : MonoBehaviour
{
    public RawImage img;
    public int capacity = 200;
    public float yMin = 0f;
    public float yMax = 1.5f;

    Texture2D tex;
    readonly List<float> values = new List<float>();

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(capacity, 64, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        img.texture = tex;
        ClearTex();
    }

    public void Push(float v)
    {
        values.Add(v);
        if (values.Count > capacity) values.RemoveAt(0);
        Redraw();
    }

    void Redraw()
    {
        ClearTex();
        if (values.Count < 2) return;

        for (int x = 0; x < values.Count; x++)
        {
            float t = Mathf.InverseLerp(yMin, yMax, values[x]);
            int y = Mathf.Clamp(Mathf.RoundToInt(t * (tex.height - 1)), 0, tex.height - 1);
            tex.SetPixel(x, y, Color.white);
            if (x > 0)
            {
                float tPrev = Mathf.InverseLerp(yMin, yMax, values[x - 1]);
                int yPrev = Mathf.Clamp(Mathf.RoundToInt(tPrev * (tex.height - 1)), 0, tex.height - 1);
                int y0 = Mathf.Min(y, yPrev), y1 = Mathf.Max(y, yPrev);
                for (int yy = y0; yy <= y1; yy++) tex.SetPixel(x, yy, Color.white);
            }
        }
        tex.Apply(false);
    }

    void ClearTex()
    {
        var cols = tex.GetPixels32();
        for (int i = 0; i < cols.Length; i++) cols[i] = new Color32(30, 30, 30, 255);
        tex.SetPixels32(cols);
        tex.Apply(false);
    }
    public void ClearSeries()
    {
        values.Clear();
        ClearTex();
    }

}
