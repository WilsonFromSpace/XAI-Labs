using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class NeuronColorKey : MonoBehaviour
{
    public ActivationHingesOverlay hinges;
    public Image sw0, sw1, sw2;
    public TMP_Text lb0, lb1, lb2;

    void Start() { Apply(); }
    public void Apply()
    {
        if (!hinges) return;
        var g = hinges.colorPerNeuron;
        if (sw0) sw0.color = g.Evaluate(0f);
        if (sw1) sw1.color = g.Evaluate(0.5f);
        if (sw2) sw2.color = g.Evaluate(1f);
        if (lb0) lb0.text = "h0"; if (lb1) lb1.text = "h1"; if (lb2) lb2.text = "h2";
    }
}
