C     DEBUG VERSION: Added print statements to trace coefficient storage
C     Original file: xnfpelsyn.for
C     Location: Lines 274-280, 361-363
C     
      CALL KAPP(N,NSTEPS,STEPWT)
      WRITE(6,15)NU,FREQ,WAVE,WAVENO
      DO 129 J=1,NRHOX
      ABTOT(J)=ACONT(J)+SIGMAC(J)
      CONTINALL(NU,J)=LOG10(ABTOT(J))
      CONTABS(NU,J)=LOG10(ACONT(J))
      CONTSCAT(NU,J)=LOG10(SIGMAC(J))
C     DEBUG: Print ACONT and stored coefficient for first few layers
      IF(J.LE.5.OR.J.EQ.NRHOX)THEN
      WRITE(6,9998)NU,J,ACONT(J),CONTABS(NU,J),LOG10(ACONT(J))
 9998 FORMAT(' DEBUG XNFPELSYN: NU=',I4,' J=',I3,' ACONT=',1PE15.8,
     1 ' CONTABS=',1PE15.8,' LOG10(ACONT)=',1PE15.8)
      ENDIF
  129 ABLOG(J)= LOG10(ABTOT(J))
      WRITE(6,105)(ABLOG(J),J=1,NRHOX)
  105 FORMAT(1X,20F5.2)
C      WRITE(10)ABLOG
C     DEBUG: Print before writing to fort.10
      IF(NU.EQ.1)THEN
      WRITE(6,9997)NU,'Writing coefficients to fort.10'
 9997 FORMAT(' DEBUG XNFPELSYN: NU=',I4,' ',A)
      ENDIF
      WRITE(10)(CONTINALL(NU,J),NU=1,1131)
      WRITE(10)(CONTABS(NU,J),NU=1,1131)
      WRITE(10)(CONTSCAT(NU,J),NU=1,1131)

