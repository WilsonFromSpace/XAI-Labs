using UnityEngine;
using TMPro;

public class ActivationStatsPanel : MonoBehaviour
{
    public TMP_Text txt;
    public float satThresh = 0.05f;   // |φ'(z)| < satThresh → saturated
    public float deadMean = 0.02f;   // mean(ReLU output) ~ 0
    public float deadVar = 0.002f;  // small variance → truly dead

    public void UpdateFrom(MLP mlp, Dataset2D data)
    {
        if (!txt || mlp == null || data == null) return;
        var X = data.XMatrix(); var Y = data.YMatrix();
        var (_, P) = mlp.Forward(X, Y);

        int N = data.count, H = mlp.Ls[0].b.Length;
        var (_, dphi) = Activations.Get(mlp.activation);

        // dZ1 = p - y
        float[,] dZ1 = new float[N, 1];
        for (int i = 0; i < N; i++) dZ1[i, 0] = P[i, 0] - Y[i, 0];

        // dA0 = dZ1 * W1^T
        var W1T = TinyTensor.Transpose(mlp.Ls[1].W);
        var dA0 = TinyTensor.MatMul(dZ1, W1T);

        // dZ0 = dA0 ⊙ φ'(Z0)
        var dphZ = TinyTensor.Apply(mlp.Ls[0].Z, dphi);
        var dZ0 = TinyTensor.Hadamard(dA0, dphZ);

        // saturation %
        int sat = 0, total = N * H;
        for (int i = 0; i < N; i++) for (int j = 0; j < H; j++) if (Mathf.Abs(dphZ[i, j]) < satThresh) sat++;

        // dead ReLU count (per unit)
        int dead = 0;
        if (mlp.activation == Act.ReLU)
        {
            for (int j = 0; j < H; j++)
            {
                float mean = 0f, var = 0f;
                for (int i = 0; i < N; i++) mean += mlp.Ls[0].A[i, j];
                mean /= N;
                for (int i = 0; i < N; i++) { float d = mlp.Ls[0].A[i, j] - mean; var += d * d; }
                var /= N;
                if (mean < deadMean && var < deadVar) dead++;
            }
        }

        // gradient flow (mean |dL/dz|)
        float gsum = 0f;
        for (int i = 0; i < N; i++) for (int j = 0; j < H; j++) gsum += Mathf.Abs(dZ0[i, j]);
        float gmean = gsum / Mathf.Max(1, total);

        txt.text = $"Saturated: {(100f * sat / Mathf.Max(1, total)):0.0}%   " +
                   (mlp.activation == Act.ReLU ? $"Dead ReLUs: {dead}/{H}   " : "") +
                   $"Mean |∂L/∂z|: {gmean:0.000}";
    }
}
