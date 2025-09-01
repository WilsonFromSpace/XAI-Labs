using UnityEngine;
using TMPro;

public class ActivationExplainerPanel : MonoBehaviour
{
    public TMP_Text txt;              // drag a TMP text here

    [Range(0.01f, 0.2f)] public float satThresh = 0.05f;
    public bool showDeadReLU = true;

    public void UpdateFrom(MLP mlp, Dataset2D data)
    {
        if (!txt || mlp == null || data == null) return;

        // --- forward to get caches ---
        var X = data.XMatrix(); var Y = data.YMatrix();
        var (loss, P) = mlp.Forward(X, Y);

        // --- simple stats used in the explanation ---
        // dZ1 = p - y
        int N = data.count, H = mlp.Ls[0].b.Length;
        float[,] dZ1 = new float[N, 1];
        for (int i = 0; i < N; i++) dZ1[i, 0] = P[i, 0] - Y[i, 0];

        // dA0 = dZ1 * W1^T
        var W1T = TinyTensor.Transpose(mlp.Ls[1].W);
        var dA0 = TinyTensor.MatMul(dZ1, W1T);

        // dZ0 = dA0 ⊙ φ'(Z0)
        var (_, dphi) = Activations.Get(mlp.activation);
        var dphZ = TinyTensor.Apply(mlp.Ls[0].Z, dphi);
        var dZ0 = TinyTensor.Hadamard(dA0, dphZ);

        // mean |dL/dz| per hidden unit
        float[] sens = new float[H];
        for (int j = 0; j < H; j++) { float s = 0f; for (int i = 0; i < N; i++) s += Mathf.Abs(dZ0[i, j]); sens[j] = s / Mathf.Max(1, N); }

        // saturated %
        int sat = 0, total = N * H;
        for (int i = 0; i < N; i++) for (int j = 0; j < H; j++) if (Mathf.Abs(dphZ[i, j]) < satThresh) sat++;
        float satPct = 100f * sat / Mathf.Max(1, total);

        // dead ReLU count (per unit)
        int dead = 0;
        if (showDeadReLU && mlp.activation == Act.ReLU)
        {
            for (int j = 0; j < H; j++)
            {
                float mean = 0f, var = 0f;
                for (int i = 0; i < N; i++) mean += mlp.Ls[0].A[i, j];
                mean /= N;
                for (int i = 0; i < N; i++) { float d = mlp.Ls[0].A[i, j] - mean; var += d * d; }
                var /= N;
                if (mean < 0.02f && var < 0.002f) dead++;
            }
        }

        int star = 0; for (int j = 1; j < H; j++) if (sens[j] > sens[star]) star = j;

        // short, beginner-friendly narration
        string actTip =
            mlp.activation == Act.Tanh ? "Tanh: symmetric, wide sensitive zone around the center line (z=0)."
          : mlp.activation == Act.Sigmoid ? "Sigmoid: narrow sensitive bands; far from them it saturates."
          : "ReLU: one-sided—active on one side of the hinge (z>0); units can go 'dead'.";

        txt.text =
$@"WHAT YOU'RE SEEING
• Each tilted line is a hidden neuron's hinge: w·x+b=0. Parallel faint bands mark where that neuron is most sensitive (high |φ′(z)|).
• Line color/label h0..h2 shows the neuron index; the number is its current influence: mean |∂L/∂z|.
• The boundary (if shown) is the network's 50/50 line.

NOW
• Activation: {mlp.activation} — {actTip}
• Saturated: {satPct:0.0}%   {(mlp.activation == Act.ReLU ? $"Dead ReLUs: {dead}/{H}   " : "")}Mean |∂L/∂z| top unit: h{star} = {sens[star]:0.000}
• Loss: {loss:0.0000}   LR: {mlp.lr:0.0000}";
    }
}
