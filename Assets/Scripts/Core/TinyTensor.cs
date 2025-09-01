using System;

public static class TinyTensor
{
    public static float[,] MatMul(float[,] A, float[,] B)
    {
        int n = A.GetLength(0), m = A.GetLength(1), p = B.GetLength(1);
        var C = new float[n, p];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < m; k++)
            {
                float aik = A[i, k];
                for (int j = 0; j < p; j++) C[i, j] += aik * B[k, j];
            }
        return C;
    }

    public static float[,] Add(float[,] A, float[,] B)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i, j] = A[i, j] + B[i, j];
        return C;
    }

    public static float[,] AddBiasRow(float[,] A, float[] b)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i, j] = A[i, j] + b[j];
        return C;
    }

    public static float[,] Apply(float[,] A, Func<float, float> f)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i, j] = f(A[i, j]);
        return C;
    }

    public static float[,] Hadamard(float[,] A, float[,] B)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var C = new float[n, m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                C[i, j] = A[i, j] * B[i, j];
        return C;
    }

    public static float[,] Transpose(float[,] A)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var T = new float[m, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                T[j, i] = A[i, j];
        return T;
    }

    public static float[] ColSum(float[,] A)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        var s = new float[m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                s[j] += A[i, j];
        return s;
    }

    public static void AddInPlace(float[,] A, float[,] B, float scale = 1f)
    {
        int n = A.GetLength(0), m = A.GetLength(1);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                A[i, j] += scale * B[i, j];
    }
    public static void AddInPlace(float[] a, float[] b, float scale = 1f)
    {
        for (int i = 0; i < a.Length; i++) a[i] += scale * b[i];
    }
}
