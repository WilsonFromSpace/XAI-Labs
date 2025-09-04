using UnityEngine;
using UnityEngine.UI;
using System;

public class ProbRailPanel : MonoBehaviour
{
    public RawImage img;
    public Color railBg = new Color(0.1f, 0.1f, 0.1f, 1f);
    public Color negCol = new Color(0.45f, 0.62f, 0.94f, 0.9f);
    public Color posCol = new Color(0.94f, 0.45f, 0.45f, 0.9f);
    public Color errCol = new Color(1f, 0.8f, 0.3f, 1f);
    public Color thrCol = Color.white;

    Texture2D tex; const int H = 70; int W = 400;

    void Awake() { if (!img) img = GetComponent<RawImage>(); Init(); }
    void Init() { tex = new Texture2D(W, H, TextureFormat.RGBA32, false); tex.wrapMode = TextureWrapMode.Clamp; img.texture = tex; }

    public void Redraw(float[,] P, float[,] Y, float thr)
    {
        if (tex == null || tex.width != W) Init();
        // bg
        var px = new Color32[W * H]; var bg = (Color32)railBg; for (int i = 0; i < px.Length; i++) px[i] = bg; tex.SetPixels32(px);

        int N = P.GetLength(0);
        int[] idx = new int[N]; for (int i = 0; i < N; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => P[a, 0].CompareTo(P[b, 0]));

        for (int k = 0; k < N; k++)
        {
            int i = idx[k];
            float p = P[i, 0]; int x = Mathf.Clamp(Mathf.RoundToInt(p * (W - 1)), 0, W - 1);
            bool isPos = Y[i, 0] > 0.5f;
            bool pred = p >= thr;
            bool correct = (pred && isPos) || (!pred && !isPos);
            Color c = correct ? (isPos ? posCol : negCol) : errCol;
            DrawTick(x, c);
        }

        int xt = Mathf.Clamp(Mathf.RoundToInt(thr * (W - 1)), 0, W - 1);
        for (int y = 0; y < H; y++) tex.SetPixel(xt, y, thrCol);

        tex.Apply(false);
    }

    void DrawTick(int x, Color c) { int y0 = 5, y1 = H - 6; for (int y = y0; y <= y1; y++) tex.SetPixel(x, y, c); }
}
