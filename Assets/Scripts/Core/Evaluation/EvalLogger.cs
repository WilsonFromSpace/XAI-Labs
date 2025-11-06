using System.Collections.Generic;
using UnityEngine;

public class EvalLogger : MonoBehaviour
{
    public static EvalLogger I { get; private set; }

    void Awake()
    {
        if (I == null) { I = this; DontDestroyOnLoad(gameObject); }
        else Destroy(gameObject);
    }

    public void Info(string evt, Dictionary<string, object> payload = null) { }
    public void ActionEvent(string evt, Dictionary<string, object> payload = null) { }
    public void Metric(string evt, Dictionary<string, object> payload = null) { }
}
