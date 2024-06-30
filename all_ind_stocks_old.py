#region imports
from AlgorithmImports import *
#endregion

ALL_IND_STOCKS_YF = \
{
  'Agric': ['CALM', 'AVO', 'CVGW', 'ALCO', 'EDBL', 'VFF', 'SANW', 'AGFY', 'NCRA', 'SEED', 'SISI', 'CEAD', 'RKDA'],
  'Food': ['MDLZ', 'KHC', 'LANC', 'PPC', 'SMPL', 'FRPT', 'JJSF', 'OTLY', 'HAIN', 'JBSS', 'BYND', 'VITL', 'AFRI', 'BRLS', 'SENEA', 'FREE', 'MAMA', 'BRID', 'LWAY', 'RGF', 'FARM', 'BSFC', 'SENEB', 'SOWG', 'RMCF', 'FTFT', 'SNAX', 'BRFH', 'STKH', 'FAMI', 'BOF', 'CHSN', 'PETZ'],
  'Soda': ['MNST', 'CCEP', 'CELH', 'COKE', 'FIZZ'],
  'Beer': ['PEP', 'KDP', 'COCO', 'WEST', 'VWE', 'WVVI', 'WVVIP', 'EAST'],
  'Smoke': ['XXII', 'ISPR', 'KAVL', 'HPCO'],
  'Toys': ['HAS', 'MAT', 'PTON', 'SONO', 'JOUT', 'FNKO', 'YYAI', 'CLAR', 'JAKK', 'ESCA', 'TRUG', 'UEIC', 'GNSS', 'BHAT', 'KOSS', 'MICS', 'IMTE'],
  'Fun': ['NFLX', 'DKNG', 'CHDN', 'IQ', 'WMG', 'BATRK', 'OSW', 'GDEN', 'SEAT', 'BATRA', 'RSVR', 'GAMB', 'GDHG', 'CSSEP', 'CDRO', 'CPHC', 'GAME', 'VTSI', 'PROP', 'HOFV', 'RDI', 'SLE', 'AGAE', 'CURI', 'GAIA', 'RDIB', 'CNVS', 'CSSE', 'BREA', 'WORX', 'CPOP'],
  'Books': ['NWSA', 'NWS', 'SCHL', 'DJCO', 'LEE', 'SOBR', 'DALN'],
  'Hshld': ['FOXF', 'IPAR', 'HELE', 'OLPX', 'IRBT', 'APOG', 'WALD', 'SNBR', 'WULF', 'GPRO', 'PRPL', 'HOFT', 'EZGO', 'ZAPP', 'FOSL', 'BSET', 'FLXS', 'VIOT', 'BRLT', 'CPSH', 'UG', 'JEWL', 'CTHR', 'SHOT', 'NVFY', 'ATER'],
  'Clths': ['CTAS', 'LULU', 'CROX', 'COLM', 'SHOO', 'GIII', 'VRA', 'SGC', 'RCKY', 'BIRD', 'MGOL', 'JRSH', 'TLF', 'SILO'],
  'Hlth': ['ACHC', 'NTRA', 'SGRY', 'OPCH', 'ENSG', 'SHC', 'GH', 'PGNY', 'LFST', 'PRVA', 'FTRE', 'AMED', 'RDNT', 'BNR', 'VCYT', 'AHCO', 'VRDN', 'ADUS', 'FLGT', 'HCSG', 'INNV', 'DCGO', 'SHCR', 'PIII', 'AIRS', 'CDNA', 'CELC', 'PNTG', 'VMD', 'CSTL', 'CMAX', 'AVAH', 'TLSI', 'QIPT', 'TALK', 'SHLT', 'LFMD', 'WGS', 'RNLX', 'OTRK', 'BIOR', 'BDSX', 'SERA', 'PSNL', 'FRES', 'DMTK', 'ACON', 'MDXH', 'MGRX', 'XGN', 'TOI', 'CNTG', 'LSTA', 'BRTX', 'EUDA', 'PMD', 'OPGN', 'BGLC', 'BACK'],
  'MedEq': ['ISRG', 'DXCM', 'GEHC', 'ALGN', 'PODD', 'COO', 'HOLX', 'MASI', 'XRAY', 'MMSI', 'NVCR', 'ICUI', 'NARI', 'IRTC', 'IART', 'INMD', 'TMDX', 'LIVN', 'AXNX', 'STAA', 'ATEC', 'ATRC', 'ESTA', 'EYE', 'PRCT', 'TNDM', 'TMCI', 'LMAT', 'UFPT', 'EMBC', 'SILK', 'SIBN', 'OM', 'RXST', 'SKIN', 'BLFS', 'KIDS', 'ATRI', 'MDXG', 'NNOX', 'OFIX', 'LQDA', 'IRMD', 'BRSH', 'TCMD', 'LUNG', 'CERS', 'RCEL', 'SRDX', 'ANGO', 'SGHT', 'AXGN', 'ARAY', 'ANIK', 'PLSE', 'OSUR', 'SMTI', 'EDAP', 'UTMD', 'PROF', 'ZIMV', 'CVRX', 'ZYXI', 'CUTR', 'INGN', 'DRTS', 'NYXH', 'OBIO', 'LYRA', 'TELA', 'BIOL', 'APYX', 'QTI', 'CTCX', 'XAIR', 'COCH', 'CLPT', 'CTSO', 'BVS', 'SMLR', 'MDAI', 'KRMD', 'DCTH', 'SLNO', 'SINT', 'INO', 'NPCE', 'HYPR', 'MGRM', 'DRIO', 'TTOO', 'LAKE', 'FONR', 'MOVE', 'INBS', 'ZJYL', 'AKLI', 'CLGN', 'LUCD', 'MHUA', 'TIVC', 'NSPR', 'PDEX', 'VANI', 'STIM', 'BEAT', 'HSCS', 'PAVM', 'ICCM', 'SRTS', 'PYPD', 'NUWE', 'DXR', 'LNSR', 'ICU', 'VVOS', 'ICAD', 'NVNO', 'BWAY', 'POCI', 'IRIX', 'CODX', 'LFWD', 'SSKN', 'RSLS', 'NMTC', 'NTRB', 'ECOR', 'MODD', 'HSDT', 'IINN', 'NDRA', 'QNRX', 'BMRA', 'LUCY', 'POAI', 'AVGR', 'STRR', 'NEPH', 'NURO', 'INVO', 'NXGL', 'CHEK', 'STSS', 'PSTV', 'VERO', 'BTCY', 'BBLG', 'BJDX', 'FEMY', 'TNON', 'NAOV', 'GCTK', 'XYLO', 'IONM', 'NXL', 'DYNT'],
  'Drugs': ['AZN', 'SNY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'IDXX', 'BNTX', 'ALNY', 'GMAB', 'ARGX', 'BGNE', 'BMRN', 'RPRX', 'LEGN', 'TECH', 'VTRS', 'APLS', 'SRPT', 'UTHR', 'NBIX', 'JAZZ', 'RGEN', 'ROIV', 'ITCI', 'GRFS', 'IONS', 'CERE', 'LNTH', 'QDEL', 'ALKS', 'PCVX', 'ASND', 'CRSP', 'NEOG', 'MDGL', 'ARWR', 'HALO', 'RVMD', 'DNLI', 'BPMC', 'ACAD', 'EVO', 'NTLA', 'RARE', 'CYTK', 'TGTX', 'FOLD', 'PRTA', 'AXSM', 'VIR', 'RIOT', 'INSM', 'KRYS', 'MLTX', 'INDV', 'BBIO', 'PTCT', 'AKRO', 'SDGR', 'XENE', 'IMCR', 'SAGE', 'AMPH', 'MORF', 'IMVT', 'GLPG', 'BEAM', 'ZLAB', 'RVNC', 'ALLR', 'NUVL', 'CVAC', 'CLDX', 'CORT', 'VTYX', 'SLRN', 'MYGN', 'HRMY', 'ZNTL', 'IOVA', 'ABCL', 'ALVO', 'HCM', 'GPCR', 'IBRX', 'SWTX', 'VCEL', 'PCRX', 'GERN', 'RCKT', 'ARVN', 'PRME', 'TVGN', 'MRUS', 'SMMT', 'IDYA', 'ETNB', 'SNDX', 'VKTX', 'DVAX', 'BCYC', 'RXRX', 'RLAY', 'IRWD', 'VRNA', 'ACLX', 'MRVI', 'SUPN', 'PTGX', 'CPRX', 'VERV', 'AGIO', 'XNCR', 'KROS', 'BCRX', 'AMLX', 'AVDL', 'REPL', 'CRNX', 'KYMR', 'OPK', 'TLRY', 'AUPH', 'SANA', 'LGND', 'COGT', 'PROK', 'IMTX', 'TVTX', 'MIRM', 'IRON', 'TWST', 'SAVA', 'ANIP', 'NVAX', 'MOR', 'PLRX', 'NAMS', 'MNKD', 'DAWN', 'RNA', 'GNLX', 'DYN', 'VALN', 'FBIOP', 'RYTM', 'ADPT', 'RGNX', 'ARQT', 'ALLO', 'VERA', 'SBFM', 'ELVN', 'FDMT', 'MESO', 'AMRX', 'TYRA', 'CYRX', 'CDMO', 'ADMA', 'PHVS', 'HLVX', 'MLYS', 'PRTC', 'LYEL', 'INVA', 'PHAR', 'TRML', 'PHAT', 'CDT', 'OCUL', 'LXRX', 'KURA', 'ARDX', 'BMEA', 'OTLK', 'SEEL', 'CRON', 'ACRS', 'ERAS', 'ARCT', 'COLL', 'MCRB', 'EWTX', 'AGEN', 'EXAI', 'SCLX', 'TARS', 'EDIT', 'CNTA', 'RAPT', 'GHRS', 'APLM', 'RNAZ', 'SPRY', 'CGEM', 'HROW', 'VYGR', 'BLUE', 'CABA', 'NRIX', 'AURA', 'AUTL', 'ZYME', 'RNAC', 'JANX', 'KNSA', 'MRNS', 'ALEC', 'ME', 'ANAB', 'TCBP', 'OCS', 'TERN', 'CMPS', 'STOK', 'SRRK', 'FATE', 'QURE', 'ZURA', 'TSVT', 'ASRT', 'ORIC', 'TRDA', 'SYRE', 'TBPH', 'OLMA', 'CHRS', 'ITOS', 'ENGN', 'AMRN', 'ALDX', 'ATXS', 'IMNM', 'AVTE', 'RPTX', 'ORGO', 'TNYA', 'FHTX', 'EOLS', 'PROC', 'CALT', 'BLTE', 'WVE', 'ENTA', 'GYRE', 'MGTX', 'ABUS', 'GLUE', 'ACRV', 'CMPX', 'ALVR', 'ESLA', 'MRSN', 'URGN', 'VNDA', 'ALXO', 'STRO', 'KALV', 'ALLK', 'DBVT', 'SVRA', 'CARM', 'CRBU', 'HUMA', 'ANNX', 'SIGA', 'SNDL', 'KOD', 'VIGL', 'SCPH', 'XERS', 'TNGX', 'INZY', 'IKNA', 'GBIO', 'CGC', 'ADVM', 'DSGN', 'MGNX', 'OMGA', 'IGMS', 'PEPG', 'AVIR', 'PMVP', 'XFOR', 'KMDA', 'OMER', 'ACIU', 'LFCR', 'NVCT', 'RGC', 'YMAB', 'PRAX', 'ACB', 'ABOS', 'IMRX', 'BDTX', 'PAHC', 'GOSS', 'ATAI', 'SLN', 'SOPH', 'IPSC', 'LSB', 'FGEN', 'IFRX', 'OPT', 'APVO', 'SGMO', 'NATR', 'LBPH', 'IMMP', 'ESPR', 'IPHA', 'EGRX', 'ANTX', 'BIVI', 'MNMD', 'ALT', 'NBTX', 'VRCA', 'FENC', 'IMAB', 'BTAI', 'ADAP', 'OVID', 'RIGL', 'KTRA', 'RLYB', 'KPTI', 'VOR', 'XOMA', 'CRMD', 'NGNE', 'CTXR', 'GRTS', 'OCEA', 'RPHM', 'IVA', 'ZVRA', 'IMUX', 'MOLN', 'JSPR', 'BTMD', 'GNFT', 'THRD', 'LRMR', 'FULC', 'SLDB', 'BGXX', 'ONCY', 'ATRA', 'AKBA', 'PDSB', 'VSTM', 'ACET', 'PRLD', 'CCCC', 'CLLS', 'QTTB', 'XBIT', 'CUE', 'MREO', 'HRTX', 'KZR', 'VERU', 'ACHV', 'RVPH', 'AADI', 'FBIO', 'INMB', 'MBIO', 'ABEO', 'LGVN', 'CERO', 'PBYI', 'NKTX', 'ATOS', 'DMAC', 'PSTX', 'TRVI', 'CYCCP', 'OGI', 'KRRO', 'CARA', 'MIST', 'TNXP', 'CAPR', 'PYXS', 'DRRX', 'STTK', 'APLT', 'XRTX', 'THTX', 'SKYE', 'JAGX', 'ATYR', 'ORMP', 'AQST', 'PRPH', 'BCAB', 'ALIM', 'BCLI', 'CRVS', 'OPTN', 'RMTI', 'OCGN', 'IVVD', 'VXRT', 'PRQR', 'GTHX', 'BLRX', 'IOBT', 'EYEN', 'HOWL', 'GLSI', 'TSHA', 'EPIX', 'TCRX', 'VCNX', 'LPTX', 'CING', 'CDXC', 'TCRT', 'CELU', 'TKNO', 'CLNN', 'THAR', 'GRI', 'BCTX', 'SCYX', 'MNOV', 'ATHA', 'CMRX', 'GLYC', 'CTMX', 'RANI', 'DTIL', 'GNTA', 'OCUP', 'NKTR', 'CNTX', 'KRON', 'CDTX', 'MYNZ', 'MYMD', 'ANIX', 'CGEN', 'XLO', 'ELTX', 'DARE', 'MDWD', 'RGLS', 'AVTX', 'AFMD', 'SLS', 'EFTR', 'CSBR', 'SLGL', 'ENTO', 'CLSD', 'SPRB', 'GALT', 'CRBP', 'CKPT', 'ETON', 'BRNS', 'ELYM', 'HCWB', 'INKT', 'SYRS', 'ASLN', 'GANX', 'APTO', 'ELEV', 'ATXI', 'IPA', 'INCR', 'FTLF', 'HOOK', 'SPRO', 'NTBL', 'RLMD', 'RZLT', 'ZVSA', 'VTVT', 'BCDA', 'TIL', 'IMMX', 'INAB', 'ASMB', 'SABS', 'PMN', 'CGTX', 'ALGS', 'TLSA', 'GNPX', 'EVAX', 'GLTO', 'CLRB', 'ADAG', 'LENZ', 'CRDF', 'NERV', 'QNCX', 'BPTH', 'COYA', 'UNCY', 'CRDL', 'CNTB', 'VYNE', 'ANEB', 'RENB', 'RNXT', 'DYAI', 'OCX', 'VIRX', 'AKAN', 'HEPA', 'PASG', 'LTRN', 'COEP', 'ALZN', 'ELDN', 'ABVC', 'BFRI', 'VBIV', 'ENLV', 'LFVN', 'LVTX', 'NRXP', 'CALC', 'BYSI', 'BOLT', 'NXTC', 'BDRX', 'CMND', 'ORGS', 'VTGN', 'UPXI', 'HUGE', 'SPRC', 'MTEM', 'TXMD', 'TARA', 'NCNA', 'ELUT', 'REVB', 'TRIB', 'NAII', 'MEIP', 'ACXP', 'CNSP', 'CRVO', 'UBX', 'MBOT', 'ACHL', 'KALA', 'AVRO', 'AWH', 'NLSP', 'AKTX', 'ICCC', 'OKYO', 'ALRN', 'RDHL', 'BNTC', 'VINC', 'CADL', 'VIRI', 'FBRX', 'PPBT', 'GDTC', 'SONN', 'KZIA', 'TFFP', 'ADTX', 'CASI', 'SNSE', 'PLUR', 'KA', 'QLI', 'AEZS', 'FLGC', 'ACST', 'ENTX', 'ABIO', 'SRZN', 'CYTO', 'VRPX', 'BFRG', 'IKT', 'SNPX', 'LUMO', 'PCSA', 'IXHL', 'SNTI', 'TRAW', 'DERM', 'MRKR', 'CDIO', 'EQ', 'LPCN', 'ONCO', 'TPST', 'NRSN', 'COCP', 'ENVB', 'ONVO', 'PMCB', 'MTEX', 'ATHE', 'CVKD', 'BON', 'BNOX', 'PALI', 'AIMD', 'DNTH', 'ADIL', 'CPIX', 'GOVX', 'ONCT', 'LSDI', 'ATNF', 'PAVS', 'GTBP', 'SHPH', 'GLMD', 'TLPH', 'ITRM', 'KPRX', 'CMMB', 'EDSA', 'NRBO', 'SNOA', 'TCON', 'APRE', 'TRVN', 'MNPR', 'INDP', 'LIPO', 'PTPI', 'HOTH', 'DRMA', 'WINT', 'EVOK', 'ADXN', 'ENSC', 'GENE', 'DRUG', 'LIXT', 'CYCC', 'PHIO', 'XCUR', 'ERNA', 'IMCC', 'IMNN', 'TTNP', 'SNGX', 'CYCN', 'IMRN', 'LEXX', 'KTTA', 'PULM', 'APM', 'GHSI', 'NEXI', 'VRAX', 'AYTU', 'SCNI', 'INM', 'UPC', 'SLRX', 'CELZ', 'QLGN', 'SMFL', 'SXTC', 'XTLB', 'SYBX', 'CRIS', 'XBIO', 'MBRX', 'PRFX'],
  'Chems': ['LIN', 'BCPC', 'MEOH', 'CSWI', 'WDFC', 'IOSP', 'GPRE', 'PCT', 'LNZA', 'BIOX', 'ORGN', 'GEVO', 'AMTX', 'CBUS', 'ALTO', 'CDXS', 'ZTEK', 'LOOP', 'SNES', 'ARQ', 'VGAS', 'EVGN', 'HGAS', 'CYTH', 'ASPI', 'GURE', 'TANH', 'CNEY', 'TOMZ', 'NITO', 'VRME'],
  'Rubbr': ['ENTG', 'NWL', 'LWLG', 'RTC', 'SWIM', 'KRT', 'DSWL', 'YHGJ', 'AREB', 'FORD'],
  'Txtls': ['TILE', 'CRWS', 'DXYN'],
  'BldMt': ['UFPI', 'RUN', 'CVCO', 'PATK', 'HLMN', 'AMWD', 'OFLX', 'AGRI', 'LEGH', 'CSTE', 'LCUT', 'SMID', 'EML', 'TPCS', 'AEHL', 'GIFI', 'RETO', 'ZKIN', 'BNRG', 'BIMI', 'FGI', 'TBLT', 'NISN', 'HIHO'],
  'Cnstr': ['FER', 'LGIH', 'MYRG', 'STRL', 'ROAD', 'IESC', 'PLPC', 'GLDD', 'BBCP', 'LSEA', 'LMB', 'MTRX', 'UHG', 'ESOA', 'CHCI', 'ADD', 'WLGS'],
  'Steel': ['STLD', 'WIRE', 'ROCK', 'MATW', 'KALU', 'GSM', 'CENX', 'ASTL', 'HAYN', 'BOOM', 'NWPX', 'AQMS', 'USAP', 'ACNT', 'BWEN', 'HUDI', 'APWC', 'OCC'],
  'FabPr': ['XPEL', 'TRS', 'PKOH', 'NTIC'],
  'Mach': ['ASML', 'LRCX', 'BKR', 'ZBRA', 'NDSN', 'LECO', 'MIDD', 'ACLS', 'CHX', 'AAON', 'WFRD', 'SYM', 'AZTA', 'ERII', 'VECO', 'KRNT', 'HSAI', 'CMCO', 'ASTE', 'ACMR', 'CRCT', 'AIRJ', 'TPIC', 'CECO', 'TWIN', 'WPRT', 'ASYS', 'KITT', 'DTI', 'AZ', 'SMX', 'NNBR', 'MNTX', 'IVAC', 'TAYD', 'PPIH', 'PFIE', 'PDYN', 'CVV', 'FTEK', 'HLP', 'NVOS', 'EKSO', 'GTEC', 'LIQT', 'ARTW'],
  'ElcEq': ['PLUG', 'WWD', 'LFUS', 'NOVT', 'FELE', 'FLNC', 'ENVX', 'NCNC', 'BLDP', 'HOLI', 'FCEL', 'EOSE', 'STI', 'WTO', 'POWL', 'MVST', 'SLDP', 'LYTS', 'NVX', 'SCWO', 'SKYX', 'HYZN', 'AMSC', 'NVVE', 'ELVA', 'BYRN', 'NEOV', 'CBAT', 'IPWR', 'DFLI', 'SOTK', 'PPSI', 'ULBI', 'CETY', 'FLUX', 'TRNR', 'ZEO', 'RAYA', 'ELBM', 'OESX', 'ADN', 'HTOO', 'XPON', 'LASE', 'POLA', 'EFOI'],
  'Autos': ['TSLA', 'PCAR', 'LI', 'FFIE', 'LCID', 'RIVN', 'IEP', 'PSNY', 'LOT', 'GNTX', 'VC', 'GT', 'VLCN', 'DOOO', 'NWTN', 'DORM', 'LAZR', 'THRM', 'UCAR', 'GTX', 'NKLA', 'PSNYW', 'NXU', 'CENN', 'GGR', 'SHYF', 'GOEV', 'BLBD', 'BLNK', 'INVZ', 'CJET', 'CVGI', 'KNDI', 'ECDA', 'NIU', 'ADSE', 'WKHS', 'MPAA', 'CAAS', 'REE', 'CPTN', 'GP', 'STRT', 'WKSP', 'XOS', 'LIDR', 'VEV', 'EVTV', 'PEV'],
  'Aero': ['HON', 'ESLT', 'AVAV', 'LILM', 'ATRO', 'HOVR', 'DPRO', 'TATT', 'MOBBW', 'AWIN', 'MOB'],
  'Ships': ['MBUU', 'MCFT', 'RVSN', 'RAIL', 'VMAR', 'VEEE', 'FRZA'],
  'Guns': ['AXON', 'RKLB', 'KTOS', 'SWBI', 'POWW', 'MNTS', 'AOUT', 'WRAP'],
  'Gold': ['PPTA', 'USGO', 'VOXR', 'HYMC', 'CHNR'],
  'Mines': ['SGML', 'USLM', 'PLL', 'CRML', 'ABAT', 'TMC', 'IONR', 'AMLI', 'EU', 'ATLX', 'LGO', 'FEAM', 'NB', 'IPX', 'SND', 'LITM', 'USAU'],
  'Coal': ['ARLP', 'METC', 'HNRG', 'METCB'],
  'Oil': ['FANG', 'APA', 'CHK', 'CHRD', 'PTEN', 'VNOM', 'ACDC', 'HPK', 'CLMT', 'DMLP', 'VTNR', 'BROG', 'BRY', 'TUSK', 'PNRG', 'KLXE', 'KGEI', 'EPSN', 'PRTG', 'DWSN', 'RCON', 'NCSM', 'USEG'],
  'Util': ['AEP', 'EXC', 'XEL', 'CEG', 'LNT', 'EVRG', 'NFE', 'NWE', 'OTTR', 'MGEE', 'AY', 'NEXT', 'ENLT', 'MSEX', 'RNW', 'CLNE', 'MNTK', 'ALCE', 'YORW', 'ARTNA', 'CWCO', 'GWRS', 'CDZI', 'PCYO', 'OPAL', 'RGCO', 'SLNG', 'VVPR', 'WAVE'],
  'Telcm': ['CMCSA', 'TMUS', 'CHTR', 'WBD', 'VOD', 'SIRI', 'FWONK', 'PARA', 'LBRDK', 'ROKU', 'FOXA', 'FOX', 'LSXMK', 'NXST', 'FYBR', 'LBTYK', 'CCOI', 'ZD', 'LSXMA', 'LBTYA', 'TIGO', 'GOGO', 'FWONA', 'VEON', 'LBRDA', 'LILAK', 'ADEA', 'SHEN', 'PARAA', 'SSP', 'ASTS', 'SBGI', 'ATEX', 'ATNI', 'CNSL', 'IHRT', 'AMCX', 'LILA', 'LSXMB', 'SPOK', 'LBTYB', 'UONEK', 'MYNA', 'SGA', 'TSAT', 'IDEX', 'UCL', 'ANGH', 'SIDU', 'BZFD', 'CMLS', 'UONE', 'NXPL', 'CXDO', 'MDIA', 'SYTA', 'OBLG', 'AYRO', 'DATS', 'BBGI'],
  'PerSv': ['CAR', 'HTZ', 'DRVN', 'LOPE', 'LAUR', 'STRA', 'UDMY', 'SDA', 'MNRO', 'EWCZ', 'PRDO', 'CRAI', 'ZCAR', 'AFYA', 'WW', 'QSG', 'EM', 'LINC', 'AREC', 'EJH', 'VERB', 'PET', 'APEI', 'JZ', 'CLEU', 'RGS', 'VSTA', 'AACG', 'YQ', 'DLPN', 'MRM', 'TCTM', 'ZCMD', 'GV', 'BOXL', 'EDTK', 'GSUN', 'EEIQ', 'XWEL', 'LXEH', 'WAFU', 'BTCT'],
  'BusSv': ['PDD', 'ADP', 'PYPL', 'NTES', 'MELI', 'ABNB', 'WDAY', 'PAYX', 'CSGP', 'VRSK', 'DASH', 'TCOM', 'EBAY', 'ICLR', 'EXAS', 'AKAM', 'GRAB', 'INCY', 'ETSY', 'WSC', 'Z', 'TTEK', 'RCM', 'MEDP', 'BILI', 'FIVN', 'EXEL', 'HQY', 'CNXC', 'EXLS', 'EXPO', 'PLTK', 'PEGA', 'AFRM', 'MARA', 'LYFT', 'FLYW', 'RELY', 'STNE', 'UXIN', 'FTAI', 'NUKK', 'PINC', 'ZG', 'ACVA', 'FTDR', 'ICFI', 'LZ', 'MGRC', 'FA', 'NEO', 'CARG', 'CRTO', 'DLO', 'NVEI', 'PAYO', 'NVEE', 'YY', 'UPBD', 'MULN', 'HEES', 'TTEC', 'CSGS', 'HURN', 'CMPR', 'ASTH', 'RILY', 'STER', 'UPWK', 'APLD', 'NRC', 'VSEC', 'ACCD', 'CCRN', 'XMTR', 'PRAA', 'TTGT', 'CIFR', 'THRY', 'STGW', 'RDWR', 'IMXI', 'ADV', 'RPAY', 'CNDT', 'IREN', 'AVXL', 'HCKT', 'OABI', 'BWMN', 'KELYA', 'BBSI', 'STBX', 'FORR', 'IIIV', 'EGHT', 'HSII', 'RGP', 'CASS', 'GDRX', 'LQDT', 'GRVY', 'QNST', 'MXCT', 'DDI', 'TRVG', 'NRDS', 'HIVE', 'RMNI', 'AERT', 'SOHU', 'RMR', 'IBEX', 'HQI', 'NUTX', 'PIXY', 'VCIG', 'XBP', 'INOD', 'LUNA', 'NCMI', 'BLDE', 'TCX', 'PRTH', 'MNY', 'PGEN', 'ANGI', 'APDN', 'III', 'BAER', 'ONMD', 'WLDN', 'OPRX', 'PMTS', 'GRPN', 'AUGX', 'PHUN', 'PFMT', 'TASK', 'VCSA', 'SY', 'BCOV', 'ABSI', 'EGIO', 'ASPS', 'VERI', 'CAUD', 'LIFW', 'MFH', 'DLHC', 'RCMT', 'HGBL', 'SURG', 'QRHC', 'PAYS', 'NOTV', 'SJ', 'TZOO', 'ARBK', 'TGL', 'KPLT', 'LTBR', 'MCHX', 'SCOR', 'FORA', 'BRAG', 'NCTY', 'EPOW', 'RSSS', 'NXTT', 'HSON', 'KELYB', 'DGHI', 'RVYL', 'SDIG', 'STCN', 'FLNT', 'GFAI', 'CHR', 'TENX', 'WKEY', 'LYT', 'IZEA', 'LICN', 'HHS', 'ANY', 'FRGT', 'KRKR', 'CISO', 'MIGI', 'SGRP', 'HTCR', 'AUUD', 'XELA', 'SWAG', 'FPAY', 'PALT', 'QH', 'MGIH', 'SAI', 'BIAF', 'SOPA', 'SLNH', 'WHLM', 'QXO', 'TCJH', 'VS', 'ISPC', 'GREE', 'PIRS', 'ALBT', 'ANTE', 'ATIF', 'DOMH', 'OLB', 'DTST', 'TC', 'ENG', 'CREG', 'LDWY', 'OCG', 'GRNQ', 'BAOS', 'MRIN', 'SWVL', 'PTIX', 'LGMK', 'ARTL', 'ONFO', 'UK', 'STAF', 'ATXG'],
  'Hardw': ['AAPL', 'CSCO', 'PANW', 'FTNT', 'NTAP', 'SMCI', 'STX', 'WDC', 'LOGI', 'FFIV', 'OMCL', 'EXTR', 'XRX', 'CRSR', 'DGII', 'SSYS', 'EVLV', 'CTLP', 'MITK', 'IMMR', 'SILC', 'RGTI', 'INVE', 'LTRX', 'RDCM', 'ALLT', 'INTZ', 'ALOT', 'QMCO', 'TACT', 'LINK', 'KTCC', 'OSS', 'HUBC', 'WLDS', 'BKYI', 'BOSC', 'SCKT'],
  'Softw': ['MSFT', 'GOOGL', 'GOOG', 'META', 'ADBE', 'INTU', 'SNPS', 'CDNS', 'BIDU', 'ADSK', 'EA', 'TTD', 'CRWD', 'CTSH', 'DDOG', 'MDB', 'ANSS', 'TEAM', 'TTWO', 'VRSN', 'ZS', 'ZM', 'PTC', 'BSY', 'SSNC', 'CHKP', 'NICE', 'JKHY', 'MANH', 'GEN', 'DOX', 'OKTA', 'OTEX', 'DOCU', 'AZPN', 'PCTY', 'ZI', 'CFLT', 'MNDY', 'SPSC', 'APP', 'CCCS', 'GLBE', 'DSGX', 'NTNX', 'CYBR', 'GTLB', 'BZ', 'DBX', 'LNW', 'SAIC', 'MSTR', 'TENB', 'DUOL', 'QLYS', 'WIX', 'ALTR', 'PYCR', 'HCP', 'FRSH', 'APPF', 'BLKB', 'MBLY', 'NCNO', 'BL', 'BRZE', 'AUR', 'WB', 'CVLT', 'FROG', 'INTA', 'VRNS', 'PRFT', 'IAS', 'CERT', 'RPD', 'SRAD', 'ALRM', 'TWKS', 'ACIW', 'JAMF', 'PRGS', 'SPT', 'HOLO', 'DJT', 'NTCT', 'MQ', 'EVCM', 'VRNT', 'AVDX', 'GDS', 'BMBL', 'TRIP', 'XTKG', 'ECX', 'APPN', 'MGNI', 'AGYS', 'MOMO', 'PDFS', 'OPRA', 'ALKT', 'PERI', 'SPNS', 'CLBT', 'KC', 'DADA', 'AILE', 'SOUN', 'DH', 'VERX', 'SABR', 'CRNC', 'DCBO', 'FORTY', 'GDEV', 'EVBG', 'RUM', 'AVPT', 'MTTR', 'CLSK', 'AMPL', 'TBLA', 'SLP', 'WKME', 'VNET', 'PUBM', 'BASE', 'BIGC', 'HSTM', 'HCAT', 'KARO', 'GDYN', 'NYAX', 'SIFY', 'VMEO', 'FAAS', 'MGIC', 'DMRC', 'PRST', 'CCSI', 'BITF', 'MYPS', 'EXFY', 'OSPN', 'DOMO', 'MTLS', 'NEXN', 'RBBN', 'RXT', 'CGNT', 'RMBL', 'LPSN', 'AIXI', 'INSE', 'TBRG', 'BNAI', 'TWOU', 'BAND', 'AMSWA', 'DOYU', 'XTIA', 'ASUR', 'KLTR', 'LDTC', 'CDLX', 'NVNI', 'API', 'SSTI', 'GMGI', 'IFBD', 'RDVT', 'BNZI', 'AISP', 'OB', 'GRRR', 'EGAN', 'ARBE', 'ARQQ', 'TRUE', 'EVER', 'TLS', 'BLZE', 'CYN', 'MLGO', 'CXAI', 'MTC', 'FNGR', 'PRCH', 'SCWX', 'XNET', 'ADTH', 'SANG', 'ISSC', 'QUBT', 'WIMI', 'UPLD', 'SMSI', 'SNCR', 'RCAT', 'MKTW', 'STRM', 'ARBB', 'MAPS', 'WFCF', 'GECC', 'DSP', 'GAN', 'GEG', 'CSPI', 'VRAR', 'GROM', 'AUID', 'AEYE', 'IPDN', 'APCX', 'IDN', 'SOGP', 'CCLD', 'DTSS', 'BMR', 'DUOT', 'MMV', 'AGMH', 'MNDO', 'AHI', 'FRSX', 'AWRE', 'ABTS', 'CREX', 'HKIT', 'CLPS', 'BCAN', 'JG', 'TAOP', 'NTWK', 'KWE', 'SVRE', 'LKCO', 'ZENV', 'LTRY', 'IVDA', 'ALAR', 'GIGM', 'HCTI', 'TSRI', 'ICLK', 'PT', 'LFLY', 'ASST', 'IDAI', 'SNAL', 'BLIN', 'GVP', 'MSGM', 'RCRT', 'DRCT', 'BLBX', 'GXAI', 'CNET', 'AMST', 'LCFY', 'MYSZ', 'WAVD'],
  'Chips': ['NVDA', 'AVGO', 'AMD', 'TXN', 'INTC', 'QCOM', 'AMAT', 'ADI', 'MU', 'NXPI', 'MRVL', 'MCHP', 'ON', 'GFS', 'MPWR', 'ENPH', 'FSLR', 'ERIC', 'SWKS', 'SEDG', 'LSCC', 'QRVO', 'ALGM', 'FLEX', 'IRDM', 'AMKR', 'RMBS', 'OLED', 'IPGP', 'NXT', 'POWI', 'VSAT', 'SLAB', 'MTSI', 'DRS', 'CRUS', 'SHLS', 'DIOD', 'AEIS', 'TSEM', 'LITE', 'CRKN', 'AMBA', 'SYNA', 'SANM', 'KLIC', 'CRDO', 'SITM', 'PLXS', 'FORM', 'MXL', 'SIMO', 'ASTI', 'VIAV', 'CSIQ', 'SATS', 'PI', 'MRCY', 'OSIS', 'HLIT', 'NVTS', 'VICR', 'UCTT', 'SPWR', 'SMTC', 'PLAB', 'INDI', 'MAXN', 'TTMI', 'SGH', 'NSSC', 'ICHR', 'COMM', 'HIMX', 'INFN', 'TYGO', 'VREX', 'AOSL', 'SNPO', 'IMOS', 'WISA', 'ADTN', 'MVIS', 'LASR', 'KE', 'CLFD', 'BELFB', 'NNDM', 'CAN', 'CEVA', 'LUNR', 'ICG', 'NVEC', 'SKYT', 'CMBM', 'AVNW', 'TROO', 'NTGR', 'SELX', 'FTCI', 'NN', 'GILT', 'LAES', 'VUZI', 'AKTS', 'MOBX', 'PWFL', 'POET', 'AUDC', 'AIP', 'CMTL', 'HEAR', 'KOPN', 'ATOM', 'TGAN', 'AAOI', 'MRAM', 'KVHI', 'ITI', 'REKR', 'CRNT', 'BEEM', 'AXTI', 'SATL', 'PEGY', 'DZSI', 'MMAT', 'GSIT', 'QUIK', 'NEON', 'BELFA', 'PXLW', 'CODA', 'MINDP', 'MIND', 'NA', 'INSG', 'CETX', 'EMKR', 'ELTK', 'AIRG', 'ONDS', 'SONM', 'LPTH', 'PRSO', 'KSCP', 'ALPP', 'RFIL', 'FKWL', 'SPI', 'EBON', 'UTSI', 'WATT', 'SPCB', 'SNT', 'NSYS', 'AMPG', 'SGMA', 'CLRO', 'LEDS', 'VISL', 'OST', 'DGLY', 'MINM', 'ASNS', 'SBET', 'MTEK'],
  'LabEq': ['KLAC', 'ROP', 'ILMN', 'TER', 'TRMB', 'BRKR', 'CGNX', 'MKSI', 'TXG', 'PACB', 'NVMI', 'ITRI', 'OLK', 'COHU', 'CAMT', 'AEHR', 'CTKB', 'QTRX', 'TRNS', 'MLAB', 'LAB', 'ALNT', 'NAUT', 'EYPT', 'BNGO', 'OPTX', 'AKYA', 'FARO', 'SEER', 'HBIO', 'MASS', 'QSI', 'SPEC', 'AXDX', 'MSAI', 'HURC', 'PRE', 'GEOS', 'FCUV', 'FEIM', 'OMIC', 'CLIR', 'TBIO', 'KEQU', 'SYPR', 'RPID', 'DAIO', 'ASTC', 'OPXS', 'AEMD', 'ELSE', 'TLIS', 'PRPO', 'THMO'],
  'Paper': ['REYN', 'PTVE', 'MLKN', 'MERC', 'VIRC', 'ILAG'],
  'Boxes': [],
  'Trans': [],
  'Whlsl': [],
  'Rtail': [],
  'Meals': [],
  'Banks': [],
  'Insur': [],
  'RlEst': [],
  'Fin': [],
  'Other': [],
}

MS_TO_FF_IND = \
  {"AGRICULTURE": ["Agric"], 
   "BUILDING_MATERIALS": ["BldMt"],
   "CHEMICALS": ["Chems"], 
   "FOREST_PRODUCTS": ["BldMt"], 
   "METALS_AND_MINING": ["Mines"], 
   "STEEL": ["Steel"], 
   "VEHICLES_AND_PARTS": ["Autos"], 
   "FURNISHINGS": ["Hshld"], 
   "FIXTURES_AND_APPLIANCES": ["Hshld"], 
   "HOMEBUILDING_AND_CONSTRUCTION": ["Cnstr"], 
   "MANUFACTURING_APPAREL_AND_ACCESSORIES": ["Clths"], 
   "PACKAGING_AND_CONTAINERS": ["Boxes"], 
   "PERSONAL_SERVICES": ["PerSv"], 
   "RESTAURANTS": ["Meals"],
   "RETAIL_CYCLICAL": ["Rtail"], 
   "TRAVEL_AND_LEISURE": ["Toys"], 
   "ASSET_MANAGEMENT": ["Fin"], 
   "BANKS": ["Banks"], 
   "CAPITAL_MARKETS": ["Fin"], 
   "INSURANCE": ["Insur"], 
   "DIVERSIFIED_FINANCIAL_SERVICES": ["Fin"], 
   "CREDIT_SERVICES": ["Fin"], 
   "REAL_ESTATE": ["RlEst"], 
   "REI_TS": ["RlEst"], 
   "BEVERAGES_ALCOHOLIC": ["Beer"], 
   "BEVERAGES_NON_ALCOHOLIC": ["Soda"], 
   "CONSUMER_PACKAGED_GOODS": ["Food"], 
   "EDUCATION": ["PerSv"], 
   "RETAIL_DEFENSIVE": ["Rtail"], 
   "TOBACCO_PRODUCTS": ["Smoke"], 
   "BIOTECHNOLOGY": ["Drugs"], 
   "DRUG_MANUFACTURERS": ["Drugs"], 
   "HEALTHCARE_PLANS": ["Hlth"], 
   "HEALTHCARE_PROVIDERS_AND_SERVICES": ["Hlth"], 
   "MEDICAL_DEVICES_AND_INSTRUMENTS": ["MedEq"], 
   "MEDICAL_DIAGNOSTICS_AND_RESEARCH": ["LabEq"], 
   "MEDICAL_DISTRIBUTION": ["MedEq"], 
   "UTILITIES_INDEPENDENT_POWER_PRODUCERS": ["Util"], 
   "UTILITIES_REGULATED": ["Util"], 
   "TELECOMMUNICATION_SERVICES": ["Telcm"], 
   "MEDIA_DIVERSIFIED": ["Books"], 
   "INTERACTIVE_MEDIA": ["Fun"], 
   "OIL_AND_GAS": ["Oil"], 
   "OTHER_ENERGY_SOURCES": ["Coal"], 
   "AEROSPACE_AND_DEFENSE": ["Aero"], 
   "BUSINESS_SERVICES": ["BusSv"], 
   "CONGLOMERATES": ["Conglomerates"], 
   "CONSTRUCTION": ["Cnstr"], 
   "FARM_AND_HEAVY_CONSTRUCTION_MACHINERY": ["Mach"], 
   "INDUSTRIAL_DISTRIBUTION": ["Whlsl"], 
   "INDUSTRIAL_PRODUCTS": ["Mach"], 
   "TRANSPORTATION": ["Trans"], 
   "WASTE_MANAGEMENT": ["Other"], 
   "SOFTWARE": ["Softw"], 
   "HARDWARE": ["Hardw"], 
   "SEMICONDUCTORS": ["Chips"]}
