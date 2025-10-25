// Assets/Scripts/Scenes/Backprop/LegendBarB.cs
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class LegendBarB : MonoBehaviour
{
    public RawImage img; public TMP_Text t0; public TMP_Text tMid; public TMP_Text t1;
    Texture2D tex; void Awake()
    {
        if (!img) img = GetComponent<RawImage>();
        tex = new Texture2D(256, 24, TextureFormat.RGBA32, false); tex.wrapMode = TextureWrapMode.Clamp; img.texture = tex;
        for (int x = 0; x < tex.width; x++)
        {
            float p = x / (tex.width - 1f);
            Color c = Color.Lerp(Color.blue, Color.red, p);
            for (int y = 0; y < tex.height; y++) tex.SetPixel(x, y, c);
        }
        // white tick at p=0.5
        int mid = tex.width / 2; for (int y = 0; y < tex.height; y++) tex.SetPixel(mid, y, Color.white);
        tex.Apply(false);
        if (t0) t0.text = "0"; if (tMid) tMid.text = "0.5"; if (t1) t1.text = "1";
    }
}
