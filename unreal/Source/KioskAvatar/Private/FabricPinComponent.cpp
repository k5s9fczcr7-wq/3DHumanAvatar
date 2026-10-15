#include "FabricPinComponent.h"

#include "Components/SkinnedMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/SkinnedAsset.h"
#include "Camera/PlayerCameraManager.h"
#include "Engine/StaticMesh.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Actor.h"
#include "Rendering/SkeletalMeshLODRenderData.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "Rendering/SkinWeightVertexBuffer.h"

namespace
{
    // MetaHumans face +Y in component space: the projection axis when there is no camera to look from.
    const FVector3f BodyForward(0.0f, 1.0f, 0.0f);

    // Möller-Trumbore: distance along Dir to the triangle, or a negative number for a miss.
    float RayTriangle(const FVector3f& Origin, const FVector3f& Dir, const FVector3f& A, const FVector3f& B,
                      const FVector3f& C, float& OutU, float& OutV)
    {
        const FVector3f E1 = B - A, E2 = C - A;
        const FVector3f P = FVector3f::CrossProduct(Dir, E2);
        const float Det = FVector3f::DotProduct(E1, P);
        if (FMath::Abs(Det) < 1e-8f)
        {
            return -1.0f;
        }
        const float Inv = 1.0f / Det;
        const FVector3f S = Origin - A;
        OutU = FVector3f::DotProduct(S, P) * Inv;
        if (OutU < 0.0f || OutU > 1.0f)
        {
            return -1.0f;
        }
        const FVector3f Q = FVector3f::CrossProduct(S, E1);
        OutV = FVector3f::DotProduct(Dir, Q) * Inv;
        if (OutV < 0.0f || OutU + OutV > 1.0f)
        {
            return -1.0f;
        }
        return FVector3f::DotProduct(E2, Q) * Inv;
    }
}

UFabricPinComponent::UFabricPinComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PostUpdateWork;   // after the garment has been animated this frame
}

void UFabricPinComponent::BeginPlay()
{
    Super::BeginPlay();
    NextBindAt = GetWorld() ? GetWorld()->GetTimeSeconds() + 0.5 : 0.0;   // let the wearer finish starting up
}

bool UFabricPinComponent::Bind()
{
    AActor* Pin = GetOwner();
    USceneComponent* Root = Pin ? Pin->GetRootComponent() : nullptr;
    AActor* Wearer = Pin ? Pin->GetAttachParentActor() : nullptr;
    if (!Root || !Wearer || !Root->GetAttachParent())
    {
        return false;
    }
    USkinnedMeshComponent* Mesh = nullptr;
    TInlineComponentArray<USkinnedMeshComponent*> Meshes(Wearer);
    for (USkinnedMeshComponent* Candidate : Meshes)
    {
        if (Candidate && Candidate->GetFName() == FabricComponentName)
        {
            Mesh = Candidate;
            break;
        }
    }
    FSkeletalMeshRenderData* RenderData = Mesh ? Mesh->GetSkeletalMeshRenderData() : nullptr;
    if (!RenderData || RenderData->LODRenderData.Num() == 0 || !Mesh->GetSkinnedAsset())
    {
        return false;
    }
    const FSkeletalMeshLODRenderData& LOD = RenderData->LODRenderData[0];
    const FPositionVertexBuffer& Positions = LOD.StaticVertexBuffers.PositionVertexBuffer;
    const FRawStaticIndexBuffer16or32Interface* Indices = LOD.MultiSizeIndexContainer.GetIndexBuffer();
    const FSkinWeightVertexBuffer* Weights = Mesh->GetSkinWeightBuffer(0);
    if (!Positions.GetVertexData() || !Indices || Indices->Num() < 3 || !Weights || !Weights->GetDataVertexBuffer()
        || !Weights->GetDataVertexBuffer()->GetWeightData() || Weights->GetNumVertices() != Positions.GetNumVertices())
    {
        UE_LOG(LogTemp, Warning, TEXT("[Pin] %s has no CPU copy of its mesh (set Allow CPU Access on LOD0); the pin stays on its bone"),
               *FabricComponentName.ToString());
        bGaveUp = true;
        return false;
    }

    // The garment as it is posed right now, and the pin where it is drawn right now (garment component space).
    const FName Socket = Root->GetAttachSocketName();
    TArray<FVector3f> Posed;
    Mesh->CacheRefToLocalMatrices(RefToLocals);
    USkinnedMeshComponent::ComputeSkinnedPositions(Mesh, Posed, RefToLocals, LOD, *Weights);
    if (Posed.Num() != static_cast<int32>(Positions.GetNumVertices()))
    {
        return false;
    }
    const FVector3f PinNow(Mesh->GetComponentTransform().InverseTransformPosition(Root->GetComponentLocation()));

    // Seat it on the fabric the camera sees behind it: cast from the camera through the pin; the first garment hit is
    // the spot it appears on. With no camera, look from straight in front of her.
    FVector3f Origin = PinNow + BodyForward * 30.0f;
    FVector3f Dir = -BodyForward;
    if (const APlayerCameraManager* Cam = UGameplayStatics::GetPlayerCameraManager(this, 0))
    {
        const FVector CamWorld = Cam->GetCameraLocation();
        if (FVector::Dist(CamWorld, Root->GetComponentLocation()) > 20.0)
        {
            Origin = FVector3f(Mesh->GetComponentTransform().InverseTransformPosition(CamWorld));
            Dir = (PinNow - Origin).GetSafeNormal();
        }
    }
    const float PinT = FVector3f::DotProduct(PinNow - Origin, Dir);
    float BestT = TNumericLimits<float>::Max(), BestU = 0.0f, BestV = 0.0f;
    int32 BestTri = INDEX_NONE;
    const int32 NumTris = Indices->Num() / 3;
    for (int32 T = 0; T < NumTris; ++T)
    {
        const FVector3f& A = Posed[Indices->Get(T * 3)];
        const FVector3f& B = Posed[Indices->Get(T * 3 + 1)];
        const FVector3f& C = Posed[Indices->Get(T * 3 + 2)];
        float U, V;
        const float Hit = RayTriangle(Origin, Dir, A, B, C, U, V);
        if (Hit > 0.0f && Hit < BestT)
        {
            BestT = Hit;
            BestTri = T;
            BestU = U;
            BestV = V;
        }
    }
    float Seat = (BestTri != INDEX_NONE) ? BestT - PinT : TNumericLimits<float>::Max();   // + = fabric behind the pin
    if (FMath::Abs(Seat) > MaxSeatDistance)
    {
        // No fabric in front of it (or too far): the nearest front-facing triangle instead.
        float BestD = TNumericLimits<float>::Max();
        BestTri = INDEX_NONE;
        for (int32 T = 0; T < NumTris; ++T)
        {
            const FVector3f& A = Posed[Indices->Get(T * 3)];
            const FVector3f& B = Posed[Indices->Get(T * 3 + 1)];
            const FVector3f& C = Posed[Indices->Get(T * 3 + 2)];
            const FVector3f N = FVector3f::CrossProduct(B - A, C - A).GetSafeNormal();
            if (FMath::Abs(FVector3f::DotProduct(N, BodyForward)) < 0.3f)
            {
                continue;
            }
            const float D = FVector3f::DistSquared((A + B + C) / 3.0f, PinNow);
            if (D < BestD)
            {
                BestD = D;
                BestTri = T;
            }
        }
        if (BestTri == INDEX_NONE)
        {
            bGaveUp = true;
            UE_LOG(LogTemp, Warning, TEXT("[Pin] no front-facing fabric on %s; the pin stays on its bone"), *FabricComponentName.ToString());
            return false;
        }
        BestU = BestV = 1.0f / 3.0f;
        Seat = FMath::Sqrt(BestD);
    }
    for (int32 K = 0; K < 3; ++K)
    {
        Tri[K] = Indices->Get(BestTri * 3 + K);
    }
    Bary = FVector3f(1.0f - BestU - BestV, BestU, BestV);
    const FVector3f A = Posed[Tri[0]], B = Posed[Tri[1]], C = Posed[Tri[2]];
    NormalSign = FVector3f::DotProduct(FVector3f::CrossProduct(B - A, C - A), BodyForward) >= 0.0f ? 1.0f : -1.0f;

    // Rest the pin on the fabric: its centre half a thickness out, minus how far it sinks in.
    float HalfThick = 1.0f;
    if (const UStaticMeshComponent* PinMesh = Cast<UStaticMeshComponent>(Root))
    {
        if (PinMesh->GetStaticMesh())
        {
            HalfThick = PinMesh->GetStaticMesh()->GetBounds().BoxExtent.Z * FMath::Abs(Root->GetComponentScale().Z);
        }
    }
    Offset = HalfThick * (1.0f - SinkFraction);

    Fabric = Mesh;
    OriginalParent = Root->GetAttachParent();
    OriginalSocket = Socket;
    OriginalRelative = Root->GetRelativeTransform();
    Root->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
    PrimaryComponentTick.AddPrerequisite(Mesh, Mesh->PrimaryComponentTick);
    bBound = true;
    UE_LOG(LogTemp, Log, TEXT("[Pin] seated on %s triangle %d (vertices %u %u %u): it hung %.1f cm %s the fabric it appears on; centre now %.2f cm off it"),
           *FabricComponentName.ToString(), BestTri, Tri[0], Tri[1], Tri[2], FMath::Abs(Seat), Seat >= 0.0f ? TEXT("in front of") : TEXT("behind"), Offset);
    return true;
}

bool UFabricPinComponent::SurfaceFrame(FVector& OutPoint, FVector& OutNormal, FVector& OutTangent)
{
    USkinnedMeshComponent* Mesh = Fabric.Get();
    FSkeletalMeshRenderData* RenderData = Mesh ? Mesh->GetSkeletalMeshRenderData() : nullptr;
    FSkinWeightVertexBuffer* Weights = Mesh ? Mesh->GetSkinWeightBuffer(0) : nullptr;
    if (!RenderData || RenderData->LODRenderData.Num() == 0 || !Weights)
    {
        return false;
    }
    const FSkeletalMeshLODRenderData& LOD = RenderData->LODRenderData[0];
    if (Tri[0] >= LOD.GetNumVertices() || Tri[1] >= LOD.GetNumVertices() || Tri[2] >= LOD.GetNumVertices())
    {
        return false;
    }
    Mesh->CacheRefToLocalMatrices(RefToLocals);
    const FTransform& ToWorld = Mesh->GetComponentTransform();
    FVector P[3];
    for (int32 K = 0; K < 3; ++K)
    {
        P[K] = ToWorld.TransformPosition(FVector(USkinnedMeshComponent::GetSkinnedVertexPosition(Mesh, Tri[K], LOD, *Weights, RefToLocals)));
    }
    OutPoint = P[0] * Bary.X + P[1] * Bary.Y + P[2] * Bary.Z;
    OutNormal = FVector::CrossProduct(P[1] - P[0], P[2] - P[0]).GetSafeNormal() * NormalSign;
    OutTangent = (P[1] - P[0]).GetSafeNormal();
    return !OutNormal.IsNearlyZero() && !OutTangent.IsNearlyZero();
}

void UFabricPinComponent::Release(const TCHAR* Why)
{
    bBound = false;
    bGaveUp = true;
    USceneComponent* Root = GetOwner() ? GetOwner()->GetRootComponent() : nullptr;
    if (Root && OriginalParent.IsValid())
    {
        Root->AttachToComponent(OriginalParent.Get(), FAttachmentTransformRules::KeepRelativeTransform, OriginalSocket);
        Root->SetRelativeTransform(OriginalRelative);
    }
    UE_LOG(LogTemp, Warning, TEXT("[Pin] back on bone %s: %s"), *OriginalSocket.ToString(), Why);
}

void UFabricPinComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    if (bGaveUp)
    {
        return;
    }
    const double Now = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0;
    if (!bBound)
    {
        if (Now >= NextBindAt && !Bind() && !bGaveUp)
        {
            NextBindAt = Now + 1.0;
            if (++BindAttempts >= 10)
            {
                bGaveUp = true;
                UE_LOG(LogTemp, Warning, TEXT("[Pin] %s not found on the actor the pin hangs on; the pin stays on its bone"),
                       *FabricComponentName.ToString());
            }
        }
        return;
    }
    FVector Point, Normal, Tangent;
    if (!SurfaceFrame(Point, Normal, Tangent))
    {
        Release(TEXT("the garment can't be read any more"));
        return;
    }
    USceneComponent* Root = GetOwner()->GetRootComponent();
    Root->SetWorldLocationAndRotation(Point + Normal * Offset, FRotationMatrix::MakeFromZX(Normal, Tangent).ToQuat());

    // First 3 minutes: log how far the original bone attachment would be from this spot (30 s windows).
    if (DriftWindows < 6 && OriginalParent.IsValid())
    {
        const FVector Old = (OriginalRelative * OriginalParent->GetSocketTransform(OriginalSocket)).GetLocation();
        const float D = FVector::Dist(Old, Point);
        if (DriftCount == 0)
        {
            DriftWindowStart = Now;
            DriftMin = DriftMax = D;
            DriftSum = 0.0f;
        }
        DriftMin = FMath::Min(DriftMin, D);
        DriftMax = FMath::Max(DriftMax, D);
        DriftSum += D;
        ++DriftCount;
        if (Now - DriftWindowStart >= 30.0)
        {
            UE_LOG(LogTemp, Log, TEXT("[Pin] last 30 s: the old bone attachment would have been %.1f-%.1f cm from the pin's spot on the fabric (mean %.1f)"),
                   DriftMin, DriftMax, DriftSum / DriftCount);
            DriftCount = 0;
            ++DriftWindows;
        }
    }
}
