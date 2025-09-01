using System;

public enum LossType { MSE, BCE }

public static class Losses
{
    public static (float loss, float[,] dY) MSE(float[,] pred, float[,] target)
    {
        int n = pred.GetLength(0), m = pred.GetLength(1);
        float L = 0f;
        var dY = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
            {
                float e = pred[i, j] - target[i, j];
                L += 0.5f * e * e;
                dY[i, j] = e;
            }
        return (L / n, dY);
    }

    // Binary cross-entropy on single output (m=1). pred should be in (0,1).
    public static (float loss, float[,] dY) BCE(float[,] pred, float[,] target)
    {
        int n = pred.GetLength(0);
        float L = 0f;
        var dY = new float[n, 1];
        for (int i = 0; i < n; i++)
        {
            float p = Math.Clamp(pred[i, 0], 1e-6f, 1f - 1e-6f);
            float y = target[i, 0];
            L += -(y * (float)Math.Log(p) + (1f - y) * (float)Math.Log(1f - p));
            dY[i, 0] = (p - y); // derivative for sigmoid + BCE combo (assuming last layer output is prob already)
        }
        return (L / n, dY);
    }
}
