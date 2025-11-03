using UnityEngine;
using UnityEngine.UI;

public class SimpleVolumeSlider : MonoBehaviour
{
    public enum Target { Music, Sfx }
    public Target target;
    public Slider slider;

    void Awake()
    {
        if (!slider) slider = GetComponent<Slider>();
        float def = 0.8f;
        float v = PlayerPrefs.GetFloat(target.ToString(), def);
        slider.SetValueWithoutNotify(v);
        Apply(v);
        slider.onValueChanged.AddListener(Apply);
    }

    void Apply(float v)
    {
        if (target == Target.Music) AudioManager.Instance?.SetMusicVolume(v);
        else AudioManager.Instance?.SetSfxVolume(v);
        PlayerPrefs.SetFloat(target.ToString(), v);
    }
}
