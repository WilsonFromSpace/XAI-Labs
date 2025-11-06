using System.Collections.Generic;
using UnityEngine;

public class ObjectiveTracker : MonoBehaviour
{
    [SerializeField] ObjectiveList objectives;
    [SerializeField] FeedbackUI feedback;
    float elapsed;
    readonly Dictionary<string, int> acts = new();
    readonly HashSet<string> tried = new();

    void Start()
    {
        foreach (var o in objectives.items)
        {
            if (ProgressService.Get(o.objectiveId) == ObjectiveStatus.Locked)
            {
                ProgressService.Set(o.objectiveId, ObjectiveStatus.Active);
                if (!string.IsNullOrEmpty(o.onActivateStory)) feedback?.Enqueue(o.onActivateStory);
            }
        }
    }

    void Update() { elapsed += Time.deltaTime; }

    // —— Call these from your scene code ——
    public void ReportFaithfulness(float F) => Check(ObjectiveType.FaithfulnessAtLeast, F >= 0 ? F : 0, (o, v) => v >= o.threshold);
    public void ReportAccuracy(float A) => Check(ObjectiveType.AccuracyAtLeast, A, (o, v) => v >= o.threshold);
    public void ReportLoss(float L) => Check(ObjectiveType.LossAtMost, L, (o, v) => v <= o.threshold);
    public void ReportAction(string key)
    {
        acts.TryGetValue(key, out var c); c++; acts[key] = c;
        foreach (var o in objectives.items)
            if (o.type == ObjectiveType.PerformActionCount && o.actionKey == key && GetStatus(o) == ObjectiveStatus.Active && c >= o.actionTarget) Complete(o);
    }
    public void ReportTriedVariant(string name)
    {
        if (tried.Add(name))
            foreach (var o in objectives.items)
                if (o.type == ObjectiveType.TryVariantsCount && GetStatus(o) == ObjectiveStatus.Active && tried.Count >= o.actionTarget) Complete(o);
    }
    public void ReportSceneFinish()
    {
        foreach (var o in objectives.items)
            if (o.type == ObjectiveType.FinishUnderSec && GetStatus(o) == ObjectiveStatus.Active && elapsed <= o.threshold) Complete(o);
    }

    void Check(ObjectiveType t, float v, System.Func<Objective, float, bool> pass)
    {
        foreach (var o in objectives.items)
            if (o.type == t && GetStatus(o) == ObjectiveStatus.Active && pass(o, v)) Complete(o);
    }

    ObjectiveStatus GetStatus(Objective o) => ProgressService.Get(o.objectiveId);

    void Complete(Objective o)
    {
        ProgressService.Set(o.objectiveId, ObjectiveStatus.Completed);
        if (!string.IsNullOrEmpty(o.onCompleteToast)) feedback?.Enqueue(o.onCompleteToast);
        EvalLogger.I?.Info("objective_complete", new() { { "id", o.objectiveId }, { "title", o.title } });
    }
}
