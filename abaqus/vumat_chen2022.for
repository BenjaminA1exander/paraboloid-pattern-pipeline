C======================================================================
C     VUMAT: Nonlinear anisotropic membrane with tension-shear coupling
C
C     Reference:
C       Chen et al. (2022) "Strength analysis of composite envelope
C       structures considering the tension-shear coupling and cutting
C       pattern effect", Thin-Walled Structures 180, 109919.
C
C     Constitutive model (TOTAL / DIRECT formulation):
C       Green-Lagrange strain E = 0.5*(U^2 - I), computed directly
C       from the right stretch tensor U provided by Abaqus.
C
C       Tension - Merged biaxial / uniaxial response surfaces:
C         Biaxial (Eq. 6):
C           S_x_bi = a1*ex + a2*ey + a3*ex^2 + a4*ey^2 + a5*ex*ey
C           S_y_bi = b1*ex + b2*ey + b3*ex^2 + b4*ey^2 + b5*ex*ey
C         Uniaxial:
C           S_x_uni = a1u*ex + a3u*ex^2
C           S_y_uni = b2u*ey + b4u*ey^2
C         Each parabola's left branch (where the slope reverses
C         past the vertex) is excluded. The reported stress is the
C         greater of the two valid branches, clamped at zero.
C
C       Shear - Tension-dependent modulus, Eq. (7)/(11):
C         G = c0 + kG*(S_x + S_y)
C         tau = G * gamma_GL     (gamma_GL = 2*E12)
C
C     No-compression treatment (combined stress / strain criterion):
C       Compute principal Green-Lagrange strains EP1 >= EP2 and
C       principal stresses SIG1 >= SIG2 (eigenvalues of the strain
C       and stress tensors).  Classify the wrinkling state from
C       both:
C       TAUT     (SIG2 > 0):              both principal stresses
C                tensile; stress passes through.
C       WRINKLED (SIG2 <= 0, EP1 > 0):    one principal stress
C                compressive but at least one principal strain
C                tensile.  Apply spectral regularisation
C                S = S+ + n*S- (compressive principal stresses
C                scaled by n=ETA_SL).
C       SLACK    (SIG1 <= 0, EP1 <= 0):   no tensile stress and no
C                tensile strain anywhere; regularised near-zero
C                stress.
C
C     Failure criterion - Eq. (19), with tension-shear coupling.
C
C     Material constants (PROPS):
C       PROPS(1)  = a1    warp linear                    (MPa)
C       PROPS(2)  = a2    weft-to-warp coupling          (MPa)
C       PROPS(3)  = a3    warp squared                   (MPa)
C       PROPS(4)  = a4    weft squared into warp         (MPa)
C       PROPS(5)  = a5    warp*weft cross                (MPa)
C       PROPS(6)  = b1    warp-to-weft coupling          (MPa)
C       PROPS(7)  = b2    weft linear                    (MPa)
C       PROPS(8)  = b3    warp squared into weft         (MPa)
C       PROPS(9)  = b4    weft squared                   (MPa)
C       PROPS(10) = b5    warp*weft cross                (MPa)
C       PROPS(11) = c0    initial shear modulus           (MPa)
C       PROPS(12) = kG    tension-shear coupling coeff
C       PROPS(13) = X     warp tensile strength           (MPa)
C       PROPS(14) = Y     weft tensile strength           (MPa)
C       PROPS(15) = S     shear strength                  (MPa)
C       PROPS(16) = alpha  interaction parameter
C       PROPS(17) = beta   interaction parameter
C       PROPS(18) = gamma  interaction parameter
C       PROPS(19) = xi     interaction parameter
C       PROPS(20) = eta    interaction parameter
C       PROPS(21) = a1u   uniaxial warp linear            (MPa)
C       PROPS(22) = a3u   uniaxial warp squared           (MPa)
C       PROPS(23) = b2u   uniaxial weft linear            (MPa)
C       PROPS(24) = b4u   uniaxial weft squared           (MPa)
C
C     State-dependent variables (SDV):
C       SDV(1) = EX        Green-Lagrange warp strain
C       SDV(2) = EY        Green-Lagrange weft strain
C       SDV(3) = GXY       engineering GL shear (2*E12)
C       SDV(4) = FF        failure factor (FF >= 1 means failed)
C       SDV(5) = G_current current shear modulus (MPa)
C       SDV(6) = WSTATE    wrinkling state (0=taut, 1=wrinkled, 2=slack)
C
C======================================================================

      SUBROUTINE VUMAT(
     1 NBLOCK, NDIR, NSHR, NSTATEV, NFIELDV, NPROPS, LANNEAL,
     2 STEPTIME, TOTALTIME, DT, CMNAME, COORDMP, CHARLENGTH,
     3 PROPS, DENSITY, STRAININC, RELSPININC,
     4 TEMPOLD, STRETCHOLD, DEFGRADOLD, FIELDOLD,
     5 STRESSOLD, STATEOLD, ENERINTERNOLD, ENERINELASOLD,
     6 TEMPNEW, STRETCHNEW, DEFGRADNEW, FIELDNEW,
     7 STRESSNEW, STATENEW, ENERINTERNNEW, ENERINELASNEW )

      INCLUDE 'VABA_PARAM.INC'

C     --- Argument arrays ---
      DIMENSION PROPS(NPROPS), DENSITY(NBLOCK)
      DIMENSION STRAININC(NBLOCK, NDIR+NSHR)
      DIMENSION RELSPININC(NBLOCK, NSHR)
      DIMENSION STRETCHNEW(NBLOCK, NDIR+NSHR)
      DIMENSION STRESSOLD(NBLOCK, NDIR+NSHR)
      DIMENSION STRESSNEW(NBLOCK, NDIR+NSHR)
      DIMENSION STATEOLD(NBLOCK, NSTATEV)
      DIMENSION STATENEW(NBLOCK, NSTATEV)
      DIMENSION ENERINTERNOLD(NBLOCK)
      DIMENSION ENERINTERNNEW(NBLOCK)
      DIMENSION ENERINELASOLD(NBLOCK)
      DIMENSION ENERINELASNEW(NBLOCK)

      CHARACTER*80 CMNAME

C     --- Local scalars ---
      REAL*8 A1, A2, A3, A4, A5
      REAL*8 B1, B2, B3, B4, B5
      REAL*8 A1U, A3U, B2U, B4U
      REAL*8 C0, KG
      REAL*8 XSTR, YSTR, SSTR
      REAL*8 FALPHA, FBETA, FGAMMA, FXI, FETA
      REAL*8 ETA_SL, CVISC
      REAL*8 CVISC_FADE_T0, CVISC_FADE_T1, CVISC_EFF
C     Wrinkling/slack residual stiffness fraction (tension-field
C     regularisation).  Raised 1e-6 -> 1e-3: at 1e-6 a wrinkled element
C     goes fully floppy, distorts in 3D and dumps its load onto the taut
C     neighbour (the hot/cold pair at the wrinkle front); 1e-3 keeps a
C     small restoring stiffness so the front stays smooth.
      PARAMETER (ETA_SL = 1.0D-3)
      PARAMETER (CVISC  = 0.01D0)
C     Bulk viscosity fade window (uses STEPTIME). Full damping for
C     STEPTIME < CVISC_FADE_T0; linearly ramped down to zero by
C     CVISC_FADE_T1; off thereafter. Set both to a huge number to
C     keep CVISC active for the entire run.
      PARAMETER (CVISC_FADE_T0 = 0.05D0)
      PARAMETER (CVISC_FADE_T1 = 0.10D0)

C     Right stretch tensor and Cauchy-Green components
      REAL*8 U11, U22, U12S
      REAL*8 CC11, CC22, CC12

C     Green-Lagrange strain
      REAL*8 EX, EY, GXY

C     Principal strain decomposition (for combined wrinkling
C     criterion)
      REAL*8 TRE, DEVE, DISC_E, EP1, EP2

C     Principal stress decomposition (for combined wrinkling
C     criterion and spectral regularisation)
      REAL*8 STR_TR, STR_DEV, STR_DSC
      REAL*8 SIG1, SIG2, SIG1M, SIG2M
      REAL*8 THSIG, CSIG, SSIG, CSIG2, SSIG2, SCSIG

C     Stress and wrinkling state
      REAL*8 SX, SY, TXY
      REAL*8 SX_OLD, SY_OLD, TXY_OLD
      REAL*8 GCUR, WSTATE

C     Failure criterion
      REAL*8 FF
      REAL*8 AX2, BY2, GXYF
      REAL*8 T1F, T2F, T3F, T4F, T5F, T6F, T7F
      REAL*8 ABSTXY

C     Energy
      REAL*8 DEX, DEY, DGXY, SPOW

C     --- Read material properties (outside loop) ---
      A1 = PROPS(1)
      A2 = PROPS(2)
      A3 = PROPS(3)
      A4 = PROPS(4)
      A5 = PROPS(5)

      B1 = PROPS(6)
      B2 = PROPS(7)
      B3 = PROPS(8)
      B4 = PROPS(9)
      B5 = PROPS(10)

      C0 = PROPS(11)
      KG = PROPS(12)

      XSTR   = PROPS(13)
      YSTR   = PROPS(14)
      SSTR   = PROPS(15)
      FALPHA = PROPS(16)
      FBETA  = PROPS(17)
      FGAMMA = PROPS(18)
      FXI    = PROPS(19)
      FETA   = PROPS(20)

      A1U = PROPS(21)
      A3U = PROPS(22)
      B2U = PROPS(23)
      B4U = PROPS(24)

C     Precompute failure criterion denominators
      AX2  = (FALPHA * XSTR)**2
      BY2  = (FBETA  * YSTR)**2
      GXYF = FGAMMA * XSTR * YSTR

C     ================================================================
C     Main loop over material points
C     ================================================================
      DO 100 I = 1, NBLOCK

C       ============================================================
C       Read right stretch tensor U (corotational, symmetric)
C       ============================================================
        U11  = STRETCHNEW(I, 1)
        U22  = STRETCHNEW(I, 2)
        U12S = STRETCHNEW(I, 4)

C       Old stresses (for energy trapezoidal rule)
        SX_OLD  = STRESSOLD(I, 1)
        SY_OLD  = STRESSOLD(I, 2)
        TXY_OLD = STRESSOLD(I, 4)

C       Strain increments (for energy computation)
        DEX  = STRAININC(I, 1)
        DEY  = STRAININC(I, 2)
        DGXY = 2.0D0 * STRAININC(I, 4)

C       Zero through-thickness strain (plane stress membrane)
        STRAININC(I, 3) = 0.0D0

C       ============================================================
C       GREEN-LAGRANGE STRAIN: E = 0.5*(U^2 - I)
C
C       Right Cauchy-Green tensor C = U^2 (U symmetric):
C         C11 = U11^2 + U12^2
C         C22 = U12^2 + U22^2
C         C12 = U11*U12 + U12*U22
C       ============================================================
        CC11 = U11*U11 + U12S*U12S
        CC22 = U12S*U12S + U22*U22
        CC12 = U11*U12S + U12S*U22

        EX  = 0.5D0*(CC11 - 1.0D0)
        EY  = 0.5D0*(CC22 - 1.0D0)
        GXY = CC12

C       ============================================================
C       PRINCIPAL GREEN-LAGRANGE STRAINS (EP1 >= EP2)
C       ============================================================
        TRE    = EX + EY
        DEVE   = EX - EY
        DISC_E = SQRT(MAX(0.25D0*DEVE*DEVE + 0.25D0*GXY*GXY,
     1                    0.0D0))
        EP1 = 0.5D0*TRE + DISC_E
        EP2 = 0.5D0*TRE - DISC_E

C       ============================================================
C       POLYNOMIAL STRESS (always evaluate first)
C       ============================================================
        CALL CHEN2022_STRESS(EX, EY,
     1    A1, A2, A3, A4, A5, B1, B2, B3, B4, B5,
     2    A1U, A3U, B2U, B4U, SX, SY)
        GCUR = C0 + KG*(SX+SY)
        TXY  = GCUR * GXY

C       ============================================================
C       PRINCIPAL STRESSES (SIG1 >= SIG2)
C       ============================================================
        STR_TR  = SX + SY
        STR_DEV = SX - SY
        STR_DSC = SQRT(MAX(0.25D0*STR_DEV*STR_DEV
     1                     + TXY*TXY, 0.0D0))
        SIG1 = 0.5D0*STR_TR + STR_DSC
        SIG2 = 0.5D0*STR_TR - STR_DSC

C       ============================================================
C       WRINKLING STATE - combined stress / strain criterion.
C       The TAUT check is non-strict (SIG2 >= 0) so that the
C       zero-strain initial state lands in TAUT with full stiffness;
C       a strict ">0" puts t=0 into SLACK, whose ETA_SL stiffness
C       is ~1000x softer and triggers an instability under load.
C         TAUT     : SIG2 >= 0
C         WRINKLED : SIG2 <  0 AND EP1 > 0
C         SLACK    : SIG2 <  0 AND EP1 <= 0  (=> SIG1 <= 0 too,
C                    no tensile strain anywhere)
C       ============================================================
        IF (SIG2 .GE. 0.0D0) THEN
          WSTATE = 0.0D0
        ELSEIF (EP1 .GT. 0.0D0) THEN
          WSTATE = 1.0D0
        ELSE
          WSTATE = 2.0D0
        ENDIF

C       ============================================================
C       WRINKLING TREATMENT
C       ============================================================
        IF (WSTATE .LT. 0.5D0) THEN
C         --- TAUT: stress passes through unchanged ---
          CONTINUE

        ELSEIF (WSTATE .LT. 1.5D0) THEN
C         --- WRINKLED: spectral decomposition of the stress
C             tensor with regularisation S = S+ + n*S-. The
C             compressive principal stress is scaled by ETA_SL;
C             the tensile principal stress passes through. ---
          THSIG = 0.0D0
          IF (ABS(TXY) .GT. 1.0D-14 .OR.
     1        ABS(STR_DEV) .GT. 1.0D-14) THEN
            THSIG = 0.5D0 * ATAN2(2.0D0*TXY, STR_DEV)
          ENDIF
          CSIG  = COS(THSIG)
          SSIG  = SIN(THSIG)
          CSIG2 = CSIG*CSIG
          SSIG2 = SSIG*SSIG
          SCSIG = SSIG*CSIG

          SIG1M = MAX(SIG1, 0.0D0) + ETA_SL*MIN(SIG1, 0.0D0)
          SIG2M = MAX(SIG2, 0.0D0) + ETA_SL*MIN(SIG2, 0.0D0)

          SX  = SIG1M*CSIG2 + SIG2M*SSIG2
          SY  = SIG1M*SSIG2 + SIG2M*CSIG2
          TXY = (SIG1M - SIG2M)*SCSIG

        ELSE
C         --- SLACK: regularised near-zero stress ---
          SX   = ETA_SL * A1 * EX
          SY   = ETA_SL * B2 * EY
          TXY  = ETA_SL * C0 * GXY
          GCUR = C0
        ENDIF

C       ============================================================
C       BULK VISCOSITY: strain-rate-dependent stress
C       Damps rapid deformation to stabilise explicit integration.
C       Faded out linearly between CVISC_FADE_T0 and CVISC_FADE_T1
C       (both in step time, seconds) so it only acts during the
C       loading transient and is removed before steady-state output.
C       ============================================================
        IF (STEPTIME .LE. CVISC_FADE_T0) THEN
          CVISC_EFF = CVISC
        ELSEIF (STEPTIME .GE. CVISC_FADE_T1) THEN
          CVISC_EFF = 0.0D0
        ELSE
          CVISC_EFF = CVISC * (CVISC_FADE_T1 - STEPTIME)
     1                       / (CVISC_FADE_T1 - CVISC_FADE_T0)
        ENDIF

        IF (DT .GT. 0.0D0 .AND. CVISC_EFF .GT. 0.0D0) THEN
          SX  = SX  + CVISC_EFF * A1 * DEX  / DT
          SY  = SY  + CVISC_EFF * B2 * DEY  / DT
          TXY = TXY + CVISC_EFF * C0 * DGXY / DT
        ENDIF


C       ============================================================
C       Assign stress components
C       ============================================================
        STRESSNEW(I, 1) = SX
        STRESSNEW(I, 2) = SY
        STRESSNEW(I, 3) = 0.0D0
        STRESSNEW(I, 4) = TXY

        IF (NSHR .GT. 1) THEN
          STRESSNEW(I, 5) = 0.0D0
          STRESSNEW(I, 6) = 0.0D0
        ENDIF

C       ============================================================
C       FAILURE CRITERION: Eq. (19)
C       Only evaluated when taut (both principal strains tensile).
C       ============================================================
        IF (WSTATE .LT. 0.5D0) THEN
          ABSTXY = ABS(TXY)
          T1F = (SX * SX) / AX2
          T2F = (SY * SY) / BY2
          T3F = (SX * SY) / GXYF
          T4F = SX / (FXI * XSTR)
          T5F = SY / (FETA * YSTR)
          T6F = (SX * ABSTXY) / (XSTR * SSTR)
          T7F = (SY * ABSTXY) / (YSTR * SSTR)
          FF = T1F + T2F - T3F + T4F + T5F + T6F + T7F
        ELSE
          FF = 0.0D0
        ENDIF

C       ============================================================
C       Store state variables
C       ============================================================
        STATENEW(I, 1) = EX
        STATENEW(I, 2) = EY
        STATENEW(I, 3) = GXY
        STATENEW(I, 4) = FF
        STATENEW(I, 5) = GCUR
        STATENEW(I, 6) = WSTATE

C       ============================================================
C       Internal energy per unit mass (trapezoidal integration)
C       ============================================================
        SPOW = 0.5D0 * (
     1    (SX_OLD  + SX)  * DEX
     2  + (SY_OLD  + SY)  * DEY
     3  + (TXY_OLD + TXY) * DGXY )

        ENERINTERNNEW(I) = ENERINTERNOLD(I) + SPOW / DENSITY(I)

 100  CONTINUE

      RETURN
      END


C======================================================================
C     CHEN2022_STRESS
C
C     Returns SX, SY as the greater of the biaxial and uniaxial
C     polynomial responses. Each parabola's left branch (where the
C     slope reverses past the vertex) is excluded. If the merged
C     response is non-positive (both branches invalid or both giving
C     negative stress), fall back to a linear regularisation
C     ETA_REG * a1 * ex (ETA_REG * b2 * ey) so the response stays
C     smooth and small instead of clamping abruptly at zero.
C
C       Biaxial:   S_x = a1*ex + a2*ey + a3*ex^2 + a4*ey^2 + a5*ex*ey
C                  S_y = b1*ex + b2*ey + b3*ex^2 + b4*ey^2 + b5*ex*ey
C       Uniaxial:  S_x = a1u*ex + a3u*ex^2
C                  S_y = b2u*ey + b4u*ey^2
C======================================================================
      SUBROUTINE CHEN2022_STRESS(EX, EY,
     1   A1, A2, A3, A4, A5, B1, B2, B3, B4, B5,
     2   A1U, A3U, B2U, B4U, SX, SY)

      INCLUDE 'VABA_PARAM.INC'

      REAL*8 EX, EY
      REAL*8 A1, A2, A3, A4, A5
      REAL*8 B1, B2, B3, B4, B5
      REAL*8 A1U, A3U, B2U, B4U
      REAL*8 SX, SY

      REAL*8 SX_BI, SY_BI, SX_UNI, SY_UNI
      REAL*8 SX_MAX, SY_MAX
      REAL*8 EX_V, EY_V
      REAL*8 NEG_INF, ETA_REG
      PARAMETER (NEG_INF = -1.0D30)
      PARAMETER (ETA_REG = 1.0D-3)

      SX_BI  = A1*EX + A2*EY + A3*EX*EX + A4*EY*EY + A5*EX*EY
      SY_BI  = B1*EX + B2*EY + B3*EX*EX + B4*EY*EY + B5*EX*EY
      SX_UNI = A1U*EX + A3U*EX*EX
      SY_UNI = B2U*EY + B4U*EY*EY

C     Drop the left branch of the biaxial sigma_x parabola in EX
      IF (A3 .GT. 1.0D-14) THEN
        EX_V = -(A1 + A5*EY) / (2.0D0*A3)
        IF (EX .LT. EX_V) SX_BI = NEG_INF
      ENDIF
C     Drop the left branch of the biaxial sigma_y parabola in EY
      IF (B4 .GT. 1.0D-14) THEN
        EY_V = -(B2 + B5*EX) / (2.0D0*B4)
        IF (EY .LT. EY_V) SY_BI = NEG_INF
      ENDIF
C     Drop the left branch of the uniaxial sigma_x parabola in EX
      IF (A3U .GT. 1.0D-14) THEN
        EX_V = -A1U / (2.0D0*A3U)
        IF (EX .LT. EX_V) SX_UNI = NEG_INF
      ENDIF
C     Drop the left branch of the uniaxial sigma_y parabola in EY
      IF (B4U .GT. 1.0D-14) THEN
        EY_V = -B2U / (2.0D0*B4U)
        IF (EY .LT. EY_V) SY_UNI = NEG_INF
      ENDIF

C     Greater of the biaxial / uniaxial branches; if non-positive
C     fall back to a linear regularisation in strain
      SX_MAX = MAX(SX_BI, SX_UNI)
      SY_MAX = MAX(SY_BI, SY_UNI)

      IF (SX_MAX .GT. 0.0D0) THEN
        SX = SX_MAX
      ELSE
        SX = ETA_REG * A1 * EX
      ENDIF

      IF (SY_MAX .GT. 0.0D0) THEN
        SY = SY_MAX
      ELSE
        SY = ETA_REG * B2 * EY
      ENDIF

      RETURN
      END