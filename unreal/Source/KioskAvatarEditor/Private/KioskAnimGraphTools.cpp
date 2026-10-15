#include "KioskAnimGraphTools.h"

#include "AnimGraphNode_ModifyCurve.h"
#include "AnimGraphNode_Root.h"
#include "Animation/AnimBlueprint.h"
#include "Animation/AnimNodeBase.h"
#include "AnimNodes/AnimNode_ModifyCurve.h"
#include "EdGraph/EdGraph.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/CompilerResultsLog.h"
#include "Kismet2/KismetEditorUtilities.h"

namespace
{
    // A node's local-space pose pin in one direction (Output Pose and Modify Curve have one of each kind they use).
    UEdGraphPin* PosePin(UEdGraphNode* Node, EEdGraphPinDirection Direction)
    {
        for (UEdGraphPin* Pin : Node->Pins)
        {
            if (Pin && Pin->Direction == Direction && Pin->PinType.PinSubCategoryObject == FPoseLink::StaticStruct())
            {
                return Pin;
            }
        }
        return nullptr;
    }

    // The node's runtime settings. UAnimGraphNode_ModifyCurve keeps them private, but they are its "Node" UPROPERTY.
    FAnimNode_ModifyCurve* Settings(UAnimGraphNode_ModifyCurve* GraphNode)
    {
        const FStructProperty* Prop = FindFProperty<FStructProperty>(UAnimGraphNode_ModifyCurve::StaticClass(), TEXT("Node"));
        return (Prop && Prop->Struct == FAnimNode_ModifyCurve::StaticStruct())
            ? Prop->ContainerPtrToValuePtr<FAnimNode_ModifyCurve>(GraphNode) : nullptr;
    }
}

FString UKioskAnimGraphTools::SetFinalCurveOffsets(UAnimBlueprint* AnimBlueprint, const TMap<FName, float>& Curves)
{
    if (!AnimBlueprint)
    {
        return TEXT("ERROR no anim blueprint");
    }
    UEdGraph* AnimGraph = nullptr;
    for (UEdGraph* Graph : AnimBlueprint->FunctionGraphs)
    {
        if (Graph && Graph->GetFName() == FName(TEXT("AnimGraph")))
        {
            AnimGraph = Graph;
            break;
        }
    }
    if (!AnimGraph)
    {
        return TEXT("ERROR no AnimGraph");
    }
    UAnimGraphNode_Root* Root = nullptr;
    for (UEdGraphNode* Node : AnimGraph->Nodes)
    {
        if ((Root = Cast<UAnimGraphNode_Root>(Node)) != nullptr)
        {
            break;
        }
    }
    UEdGraphPin* RootIn = Root ? PosePin(Root, EGPD_Input) : nullptr;
    if (!RootIn || RootIn->LinkedTo.Num() != 1)
    {
        return FString::Printf(TEXT("ERROR Output Pose not found or not linked to exactly one node (%d)"),
                               RootIn ? RootIn->LinkedTo.Num() : -1);
    }
    UEdGraphPin* Upstream = RootIn->LinkedTo[0];

    // Ours = a Modify Curve in Add mode feeding Output Pose directly.
    UAnimGraphNode_ModifyCurve* Ours = Cast<UAnimGraphNode_ModifyCurve>(Upstream->GetOwningNode());
    if (Ours && (!Settings(Ours) || Settings(Ours)->ApplyMode != EModifyCurveApplyMode::Add))
    {
        Ours = nullptr;
    }

    FString Done;
    if (Curves.Num() == 0)
    {
        if (!Ours)
        {
            return TEXT("nothing to remove");
        }
        UEdGraphPin* OursIn = PosePin(Ours, EGPD_Input);
        UEdGraphPin* Before = (OursIn && OursIn->LinkedTo.Num() == 1) ? OursIn->LinkedTo[0] : nullptr;
        AnimGraph->RemoveNode(Ours);
        if (Before)
        {
            Before->MakeLinkTo(RootIn);
        }
        Done = TEXT("removed");
    }
    else if (Ours)
    {
        Settings(Ours)->CurveMap = Curves;
        Done = TEXT("updated");
    }
    else
    {
        FGraphNodeCreator<UAnimGraphNode_ModifyCurve> Creator(*AnimGraph);
        UAnimGraphNode_ModifyCurve* Added = Creator.CreateNode(false);
        FAnimNode_ModifyCurve* AddedSettings = Settings(Added);
        if (!AddedSettings)
        {
            Creator.Finalize();
            AnimGraph->RemoveNode(Added);
            return TEXT("ERROR cannot reach the Modify Curve node's settings");
        }
        AddedSettings->ApplyMode = EModifyCurveApplyMode::Add;
        AddedSettings->Alpha = 1.0f;
        AddedSettings->CurveMap = Curves;
        Added->NodePosX = Root->NodePosX - 320;
        Added->NodePosY = Root->NodePosY + 160;
        Added->NodeComment = TEXT("Added on top of everything (KioskAnimGraphTools::SetFinalCurveOffsets)");
        Added->bCommentBubbleVisible = true;
        Creator.Finalize();
        UEdGraphPin* AddedIn = PosePin(Added, EGPD_Input);
        UEdGraphPin* AddedOut = PosePin(Added, EGPD_Output);
        if (!AddedIn || !AddedOut)
        {
            AnimGraph->RemoveNode(Added);
            return TEXT("ERROR the new Modify Curve node has no pose pins");
        }
        RootIn->BreakLinkTo(Upstream);
        Upstream->MakeLinkTo(AddedIn);
        AddedOut->MakeLinkTo(RootIn);
        Done = TEXT("inserted");
    }

    FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(AnimBlueprint);
    FCompilerResultsLog Results;
    FKismetEditorUtilities::CompileBlueprint(AnimBlueprint, EBlueprintCompileOptions::None, &Results);
    if (AnimBlueprint->Status == BS_Error)
    {
        return FString::Printf(TEXT("ERROR %s, but the blueprint failed to compile (%d errors)"), *Done, Results.NumErrors);
    }
    AnimBlueprint->MarkPackageDirty();
    return FString::Printf(TEXT("%s: %d curve offsets, compiled with %d warnings"), *Done, Curves.Num(), Results.NumWarnings);
}

namespace
{
    const TCHAR* EyeCurveNames[] = {
        TEXT("CTRL_expressions_eyeLookLeftL"), TEXT("CTRL_expressions_eyeLookRightL"),
        TEXT("CTRL_expressions_eyeLookUpL"), TEXT("CTRL_expressions_eyeLookDownL"),
        TEXT("CTRL_expressions_eyeLookLeftR"), TEXT("CTRL_expressions_eyeLookRightR"),
        TEXT("CTRL_expressions_eyeLookUpR"), TEXT("CTRL_expressions_eyeLookDownR")};

    bool IsEyeNode(UEdGraphNode* Node)
    {
        UAnimGraphNode_ModifyCurve* M = Cast<UAnimGraphNode_ModifyCurve>(Node);
        FAnimNode_ModifyCurve* S = M ? Settings(M) : nullptr;
        return S && S->ApplyMode == EModifyCurveApplyMode::Add && S->CurveMap.Contains(FName(EyeCurveNames[0]));
    }
}

FString UKioskAnimGraphTools::SetEyeCurveNode(UAnimBlueprint* AnimBlueprint, bool bInstall)
{
    if (!AnimBlueprint)
    {
        return TEXT("ERROR no anim blueprint");
    }
    UEdGraph* AnimGraph = nullptr;
    for (UEdGraph* Graph : AnimBlueprint->FunctionGraphs)
    {
        if (Graph && Graph->GetFName() == FName(TEXT("AnimGraph")))
        {
            AnimGraph = Graph;
            break;
        }
    }
    if (!AnimGraph)
    {
        return TEXT("ERROR no AnimGraph");
    }
    UAnimGraphNode_Root* Root = nullptr;
    for (UEdGraphNode* Node : AnimGraph->Nodes)
    {
        if ((Root = Cast<UAnimGraphNode_Root>(Node)) != nullptr)
        {
            break;
        }
    }
    UEdGraphPin* RootIn = Root ? PosePin(Root, EGPD_Input) : nullptr;
    if (!RootIn || RootIn->LinkedTo.Num() != 1)
    {
        return TEXT("ERROR Output Pose not found or not linked to exactly one node");
    }

    // Insert point: in front of the final offsets node (the resting smile) if there is one, else Output Pose.
    UEdGraphPin* Into = RootIn;
    UEdGraphNode* Final = RootIn->LinkedTo[0]->GetOwningNode();
    UAnimGraphNode_ModifyCurve* FinalCurve = Cast<UAnimGraphNode_ModifyCurve>(Final);
    if (FinalCurve && Settings(FinalCurve) && Settings(FinalCurve)->ApplyMode == EModifyCurveApplyMode::Add && !IsEyeNode(Final))
    {
        Into = PosePin(FinalCurve, EGPD_Input);
    }
    if (!Into || Into->LinkedTo.Num() != 1)
    {
        return TEXT("ERROR the insert point is not linked to exactly one node");
    }
    UEdGraphPin* Upstream = Into->LinkedTo[0];
    UEdGraphNode* Existing = IsEyeNode(Upstream->GetOwningNode()) ? Upstream->GetOwningNode() : nullptr;

    FString Done;
    if (!bInstall)
    {
        if (!Existing)
        {
            return TEXT("nothing to remove");
        }
        UEdGraphPin* ExistingIn = PosePin(Existing, EGPD_Input);
        UEdGraphPin* Before = (ExistingIn && ExistingIn->LinkedTo.Num() == 1) ? ExistingIn->LinkedTo[0] : nullptr;
        AnimGraph->RemoveNode(Existing);
        if (Before)
        {
            Before->MakeLinkTo(Into);
        }
        Done = TEXT("removed");
    }
    else if (Existing)
    {
        Done = TEXT("already there");
    }
    else
    {
        FGraphNodeCreator<UAnimGraphNode_ModifyCurve> Creator(*AnimGraph);
        UAnimGraphNode_ModifyCurve* Added = Creator.CreateNode(false);
        FAnimNode_ModifyCurve* AddedSettings = Settings(Added);
        if (!AddedSettings)
        {
            Creator.Finalize();
            AnimGraph->RemoveNode(Added);
            return TEXT("ERROR cannot reach the Modify Curve node's settings");
        }
        AddedSettings->ApplyMode = EModifyCurveApplyMode::Add;
        AddedSettings->Alpha = 1.0f;
        for (const TCHAR* Name : EyeCurveNames)
        {
            AddedSettings->CurveMap.Add(FName(Name), 0.0f);
        }
        Added->NodePosX = Root->NodePosX - 640;
        Added->NodePosY = Root->NodePosY + 160;
        Added->NodeComment = TEXT("Per-eye gaze: values written every frame by UEyeGazeComponent (KioskAnimGraphTools::SetEyeCurveNode)");
        Added->bCommentBubbleVisible = true;
        Creator.Finalize();
        UEdGraphPin* AddedIn = PosePin(Added, EGPD_Input);
        UEdGraphPin* AddedOut = PosePin(Added, EGPD_Output);
        if (!AddedIn || !AddedOut)
        {
            AnimGraph->RemoveNode(Added);
            return TEXT("ERROR the new Modify Curve node has no pose pins");
        }
        Into->BreakLinkTo(Upstream);
        Upstream->MakeLinkTo(AddedIn);
        AddedOut->MakeLinkTo(Into);
        Done = TEXT("inserted");
    }

    FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(AnimBlueprint);
    FCompilerResultsLog Results;
    FKismetEditorUtilities::CompileBlueprint(AnimBlueprint, EBlueprintCompileOptions::None, &Results);
    if (AnimBlueprint->Status == BS_Error)
    {
        return FString::Printf(TEXT("ERROR %s, but the blueprint failed to compile (%d errors)"), *Done, Results.NumErrors);
    }
    AnimBlueprint->MarkPackageDirty();
    return FString::Printf(TEXT("%s (in front of %s), compiled with %d warnings"), *Done,
                           Into == RootIn ? TEXT("Output Pose") : TEXT("the resting-smile node"), Results.NumWarnings);
}
