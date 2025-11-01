using UnityEngine;
using UnityEngine.Audio;
using System.Collections;
using System.Collections.Generic;

public class AudioManager : MonoBehaviour
{
    public static AudioManager Instance { get; private set; }

    [Header("Mixer (optional)")]
    public AudioMixer mixer;                 // assign in Inspector
    [Tooltip("Name of exposed parameter for music volume in dB")]
    public string musicParam = "MusicVolume";
    [Tooltip("Name of exposed parameter for sfx volume in dB")]
    public string sfxParam = "SFXVolume";

    [Header("Music Sources (2 for crossfade)")]
    public AudioSource musicA;               // assign in Inspector
    public AudioSource musicB;               // assign in Inspector
    [Range(0f, 5f)] public float defaultFadeTime = 0.75f;

    [Header("SFX")]
    public AudioSource sfxSource;            // simple route: one-shot source
    [Tooltip("Optional: small pool for overlapping UI sounds")]
    public int sfxPoolSize = 4;
    private List<AudioSource> sfxPool;

    private bool usingA = true;
    private Coroutine musicFadeRoutine;

    void Awake()
    {
        // Simple, safe singleton that survives scene loads
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);             // avoid duplicates between scenes
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        // Build SFX pool if requested
        if (sfxPoolSize > 1)
        {
            sfxPool = new List<AudioSource>(sfxPoolSize);
            sfxPool.Add(sfxSource);
            for (int i = 1; i < sfxPoolSize; i++)
            {
                var extra = gameObject.AddComponent<AudioSource>();
                extra.outputAudioMixerGroup = sfxSource.outputAudioMixerGroup;
                extra.playOnAwake = false;
                sfxPool.Add(extra);
            }
        }
    }

    // -------- MUSIC --------

    public void PlayMusic(AudioClip clip, float fadeTime = -1f, bool loop = true, float pitch = 1f)
    {
        if (clip == null) return;
        fadeTime = fadeTime < 0f ? defaultFadeTime : fadeTime;

        AudioSource from = usingA ? musicA : musicB;
        AudioSource to = usingA ? musicB : musicA;

        to.clip = clip;
        to.loop = loop;
        to.pitch = pitch;
        to.volume = 0f;
        to.Play();

        if (musicFadeRoutine != null) StopCoroutine(musicFadeRoutine);
        musicFadeRoutine = StartCoroutine(Crossfade(from, to, fadeTime));
        usingA = !usingA;
    }

    public void StopMusic(float fadeTime = -1f)
    {
        fadeTime = fadeTime < 0f ? defaultFadeTime : fadeTime;
        if (musicFadeRoutine != null) StopCoroutine(musicFadeRoutine);
        var aOn = musicA.isPlaying ? musicA : null;
        var bOn = musicB.isPlaying ? musicB : null;
        StartCoroutine(FadeOutThenStop(aOn, fadeTime));
        StartCoroutine(FadeOutThenStop(bOn, fadeTime));
    }

    private IEnumerator Crossfade(AudioSource from, AudioSource to, float t)
    {
        float time = 0f;
        float fromStart = from ? from.volume : 0f;
        while (time < t)
        {
            float k = time / t;
            if (to) to.volume = Mathf.Lerp(0f, 1f, k);
            if (from) from.volume = Mathf.Lerp(fromStart, 0f, k);
            time += Time.unscaledDeltaTime;
            yield return null;
        }
        if (to) to.volume = 1f;
        if (from)
        {
            from.volume = 0f;
            from.Stop();
            from.clip = null;
        }
    }

    private IEnumerator FadeOutThenStop(AudioSource src, float t)
    {
        if (src == null) yield break;
        float start = src.volume;
        float time = 0f;
        while (time < t)
        {
            float k = time / t;
            src.volume = Mathf.Lerp(start, 0f, k);
            time += Time.unscaledDeltaTime;
            yield return null;
        }
        src.Stop();
        src.clip = null;
        src.volume = 0f;
    }

    // -------- SFX --------

    public void PlaySFX(AudioClip clip, float volume = 1f, float pitch = 1f)
    {
        if (clip == null) return;

        AudioSource src = sfxPoolSize > 1 ? GetFreeSfxSource() : sfxSource;
        if (src == null) return;

        src.pitch = pitch;
        src.PlayOneShot(clip, volume);
    }

    private AudioSource GetFreeSfxSource()
    {
        for (int i = 0; i < sfxPool.Count; i++)
        {
            if (!sfxPool[i].isPlaying) return sfxPool[i];
        }
        // all busy—just reuse the first
        return sfxPool[0];
    }

    // -------- Volume Helpers (Mixer expects dB) --------

    public void SetMusicVolume01(float linear) => SetDb(musicParam, LinearToDb(linear));
    public void SetSfxVolume01(float linear) => SetDb(sfxParam, LinearToDb(linear));

    private void SetDb(string param, float db)
    {
        if (mixer && !string.IsNullOrEmpty(param))
            mixer.SetFloat(param, db);
    }

    private float LinearToDb(float linear)
    {
        // clamp very low values to avoid -Inf
        linear = Mathf.Clamp(linear, 0.0001f, 1f);
        return Mathf.Log10(linear) * 20f; // 0..1 -> -80..0-ish
    }
}
