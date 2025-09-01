using UnityEngine;
using System.Collections.Generic;

/// Draws the p(x)=0.5 isocontour using marching squares (no textures).
[ExecuteAlways]
public class DecisionBoundaryCurve : MonoBehaviour
{
    public Material lineMat;                  // URP/Unlit, ZTest Always (nice)
    public Vector2 worldMin = new(-5, -5);
    public Vector2 worldMax = new(5, 5);
    public int grid = 96;
    public float threshold = 0.5f;
    public float lineWidth = 2f;

    readonly List<Vector3> segs = new();     // pairs of points (A,B,A,B,...)

    public void Redraw(System.Func<Vector2, float> prob)
    {
        segs.Clear();
        if (prob == null || grid < 2) return;

        int nx = grid, ny = grid;
        float dx = (worldMax.x - worldMin.x) / (nx - 1);
        float dy = (worldMax.y - worldMin.y) / (ny - 1);

        // sample scalar field
        float[,] f = new float[nx, ny];
        for (int iy = 0; iy < ny; iy++)
            for (int ix = 0; ix < nx; ix++)
            {
                var p = new Vector2(worldMin.x + ix * dx, worldMin.y + iy * dy);
                f[ix, iy] = prob(p);
            }

        // marching squares per cell
        for (int iy = 0; iy < ny - 1; iy++)
            for (int ix = 0; ix < nx - 1; ix++)
            {
                // corners: (ix,iy)=A, (ix+1,iy)=B, (ix+1,iy+1)=C, (ix,iy+1)=D
                float FA = f[ix, iy], FB = f[ix + 1, iy], FC = f[ix + 1, iy + 1], FD = f[ix, iy + 1];
                int m = 0;
                if (FA > threshold) m |= 1; if (FB > threshold) m |= 2; if (FC > threshold) m |= 4; if (FD > threshold) m |= 8;
                if (m == 0 || m == 15) continue;

                Vector2 A = new(worldMin.x + ix * dx, worldMin.y + iy * dy);
                Vector2 B = new(worldMin.x + (ix + 1) * dx, worldMin.y + iy * dy);
                Vector2 C = new(worldMin.x + (ix + 1) * dx, worldMin.y + (iy + 1) * dy);
                Vector2 D = new(worldMin.x + ix * dx, worldMin.y + (iy + 1) * dy);

                Vector2 E(float t, Vector2 P, Vector2 Q) => Vector2.Lerp(P, Q, t);
                float T(float f0, float f1) { float denom = (f1 - f0); return Mathf.Approximately(denom, 0) ? 0.5f : (threshold - f0) / denom; }

                // edge interpolation
                Vector2 eAB = E(T(FA, FB), A, B);
                Vector2 eBC = E(T(FB, FC), B, C);
                Vector2 eCD = E(T(FC, FD), C, D);
                Vector2 eDA = E(T(FD, FA), D, A);

                // cases (representative pairs)
                void Add(Vector2 P, Vector2 Q) { segs.Add(P); segs.Add(Q); }

                switch (m)
                {
                    case 1: case 14: Add(eAB, eDA); break;
                    case 2: case 13: Add(eAB, eBC); break;
                    case 3: case 12: Add(eBC, eDA); break;
                    case 4: case 11: Add(eBC, eCD); break;
                    case 5: Add(eAB, eBC); Add(eCD, eDA); break; // saddle
                    case 6: case 9: Add(eAB, eCD); break;
                    case 7: case 8: Add(eCD, eDA); break;
                    case 10: Add(eAB, eDA); Add(eBC, eCD); break; // saddle
                }
            }
    }

    void OnRenderObject()
    {
        if (lineMat == null || segs.Count == 0) return;
        lineMat.SetPass(0);
        GL.PushMatrix();
        GL.MultMatrix(Matrix4x4.identity);
        GL.Begin(GL.LINES);
        GL.Color(Color.white);
        foreach (var v in segs) GL.Vertex3(v.x, v.y, 0);
        GL.End();
        GL.PopMatrix();
    }
}
