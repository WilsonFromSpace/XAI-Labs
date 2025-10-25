using UnityEngine;
using System;

public static class DataFactory2D
{
    static System.Random rnd = new System.Random();

    public static void MakeBlobs(int nTotal, float spread, float overlap, float rotDeg,
                                 out Vector2[] pts, out int[] y)
    {
        int n = Mathf.Max(2, nTotal);
        pts = new Vector2[n]; y = new int[n];

        // centers move together as overlap increases (0 = well separated, 1 = on top)
        float d = Mathf.Lerp(1.6f, 0.1f, Mathf.Clamp01(overlap));
        Vector2 c0 = new(-d, 0f), c1 = new(d, 0f);

        for (int i = 0; i < n; i++)
        {
            bool cls1 = (i % 2 == 1);
            Vector2 c = cls1 ? c1 : c0;
            pts[i] = c + spread * RandN2();
            y[i] = cls1 ? 1 : 0;
        }
        Rotate(pts, rotDeg);
        NormalizeExtent(pts, 1.2f); // keep in view
    }

    public static void MakeMoons(int nTotal, float noise, float gap, float rotDeg,
                                 out Vector2[] pts, out int[] y)
    {
        int n = Mathf.Max(2, nTotal);
        pts = new Vector2[n]; y = new int[n];
        int nHalf = n / 2;

        // upper arc (class 1)
        for (int i = 0; i < nHalf; i++)
        {
            float t = (i / (float)(nHalf - 1)) * Mathf.PI;
            Vector2 p = new(Mathf.Cos(t), Mathf.Sin(t));
            p += noise * RandN2();
            pts[i] = p; y[i] = 1;
        }
        // lower arc shifted
        for (int i = 0; i < n - nHalf; i++)
        {
            float t = (i / (float)(n - nHalf - 1)) * Mathf.PI;
            Vector2 p = new(Mathf.Cos(t), -Mathf.Sin(t));
            p += new Vector2(1.0f - gap, -0.5f + gap); // control overlap via gap
            p += noise * RandN2();
            pts[nHalf + i] = p; y[nHalf + i] = 0;
        }
        Rotate(pts, rotDeg);
        NormalizeExtent(pts, 1.15f);
    }

    public static void MakeRings(int nTotal, float noise, float gap, float rotDeg,
                                 out Vector2[] pts, out int[] y)
    {
        int n = Mathf.Max(2, nTotal);
        pts = new Vector2[n]; y = new int[n];
        int nHalf = n / 2;

        float r0 = 0.6f;
        float r1 = r0 + Mathf.Lerp(0.15f, 0.45f, Mathf.Clamp01(gap)); // bigger gap -> wider ring separation

        for (int i = 0; i < nHalf; i++)
        {
            float t = UnityEngine.Random.value * 2f * Mathf.PI;
            Vector2 p = Polar(r0, t) + noise * RandN2();
            pts[i] = p; y[i] = 0;
        }
        for (int i = 0; i < n - nHalf; i++)
        {
            float t = UnityEngine.Random.value * 2f * Mathf.PI;
            Vector2 p = Polar(r1, t) + noise * RandN2();
            pts[nHalf + i] = p; y[nHalf + i] = 1;
        }
        Rotate(pts, rotDeg);
        NormalizeExtent(pts, 1.15f);
    }

    // --- helpers ---
    static Vector2 RandN2()
    {
        // Box-Muller
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        float g = (float)(Mathf.Sqrt(-2f * Mathf.Log((float)u1)) * Mathf.Cos(2f * Mathf.PI * (float)u2));
        double v1 = 1.0 - rnd.NextDouble();
        double v2 = 1.0 - rnd.NextDouble();
        float h = (float)(Mathf.Sqrt(-2f * Mathf.Log((float)v1)) * Mathf.Sin(2f * Mathf.PI * (float)v2));
        return new Vector2(g, h);
    }

    static Vector2 Polar(float r, float t) => new(r * Mathf.Cos(t), r * Mathf.Sin(t));

    static void Rotate(Vector2[] pts, float deg)
    {
        float a = deg * Mathf.Deg2Rad;
        float ca = Mathf.Cos(a), sa = Mathf.Sin(a);
        for (int i = 0; i < pts.Length; i++)
        {
            var p = pts[i];
            pts[i] = new Vector2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
        }
    }

    static void NormalizeExtent(Vector2[] pts, float targetHalfSpan)
    {
        float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
        foreach (var p in pts) { if (p.x < minx) minx = p.x; if (p.y < miny) miny = p.y; if (p.x > maxx) maxx = p.x; if (p.y > maxy) maxy = p.y; }
        Vector2 c = new((minx + maxx) / 2f, (miny + maxy) / 2f);
        float span = Mathf.Max(maxx - minx, maxy - miny);
        float s = span > 1e-6f ? (2f * targetHalfSpan) / span : 1f;
        for (int i = 0; i < pts.Length; i++) pts[i] = (pts[i] - c) * s;
    }
}
