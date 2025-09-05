using UnityEngine;
using UnityEngine.UI;

public class DropoutRainPanel : MonoBehaviour
{
    public RawImage img;
    public Color bg = new(0, 0, 0, 0), drop = new(0.9f, 0.9f, 1f, 0.7f);
    Texture2D tex; const int W = 420, H = 80; System.Random r = new System.Random();

    void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(W, H, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        img.texture = tex;
    }

    public void Redraw(MLP_Capacity mlp)
    {
        var px = new Color32[W * H]; var bgc = (Color32)bg; for (int i = 0; i < px.Length; i++) px[i] = bgc; tex.SetPixels32(px);
        int Hn = mlp.Ls[0].b.Length;
        int drops = Mathf.RoundToInt(Mathf.Clamp01(mlp.dropoutP) * Mathf.Max(1, Hn));
        for (int i = 0; i < drops; i++) { int x = r.Next(W); for (int y = H - 1; y >= 0; y--) tex.SetPixel(x, y, drop); }
        tex.Apply(false);
    }
}
