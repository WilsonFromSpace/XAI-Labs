using UnityEngine;
using UnityEngine.UI;

public class VolumeSlider : MonoBehaviour
{
    public enum Channel { Music, SFX }
    public Channel channel;
    public Slider slider;

    void Awake()
    {
        if (!slider) slider = GetComponent<Slider>();
        // Optional: set default UI value (0..1). You can load from PlayerPrefs here.
        if (!PlayerPrefs.HasKey(channel.ToString()))
            PlayerPrefs.SetFloat(channel.ToString(), 0.8f);

        float v = PlayerPrefs.GetFloat(channel.ToString(), 0.8f);
        slider.SetValueWithoutNotify(v);
        Apply(v);
        slider.onValueChanged.AddListener(Apply);
    }

    void Apply(float v)
    {
        if (channel == Channel.Music) AudioManager.Instance?.SetMusicVolume01(v);
        else AudioManager.Instance?.SetSfxVolume01(v);
        PlayerPrefs.SetFloat(channel.ToString(), v);
    }
}
