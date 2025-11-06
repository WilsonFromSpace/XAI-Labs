using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable] public class ObjState { public string id; public string status; public string ts; }
[Serializable] public class ProgressModel { public List<ObjState> states = new(); }

public static class ProgressService
{
    static string PathFile => Path.Combine(Application.persistentDataPath, "progress.json");
    static ProgressModel _cache;

    public static ProgressModel Load()
    {
        if (_cache != null) return _cache;
        if (File.Exists(PathFile)) _cache = JsonUtility.FromJson<ProgressModel>(File.ReadAllText(PathFile));
        if (_cache == null) _cache = new ProgressModel();
        return _cache;
    }

    public static ObjectiveStatus Get(string id)
    {
        var s = Load().states.Find(x => x.id == id);
        return s == null ? ObjectiveStatus.Locked : (s.status == "Completed" ? ObjectiveStatus.Completed : ObjectiveStatus.Active);
    }

    public static void Set(string id, ObjectiveStatus st)
    {
        var p = Load();
        var s = p.states.Find(x => x.id == id);
        if (s == null) { s = new ObjState { id = id }; p.states.Add(s); }
        s.status = st.ToString(); s.ts = DateTime.UtcNow.ToString("o");
        File.WriteAllText(PathFile, JsonUtility.ToJson(p, true));
    }

    public static (int completed, int total) CountForScene(string scene)
    {
        var p = Load();
        int total = 0, done = 0;
        foreach (var s in p.states)
        {
            if (!s.id.StartsWith(scene + "_")) continue;
            total++; if (s.status == "Completed") done++;
        }
        return (done, total);
    }
}
