### CAS_KPBMA_MAP3K5_IC50s

Ligand Number Names SMILES sheet

| 열 이름 | 설명 |
| --- | --- |
| Ligand Number | 각 화합물의 참조 번호. .svg 파일 등 시각화와 연결 |
| Substance Name | 화합물 이름 (예: IUPAC 또는 일반명) |
| SMILES | 화학 구조 정보 (SMILES 형식) |
| Stereo Attribute Note | 입체화학(stereochemistry) 관련 주석 |

MAP3K5 Ligand IC50s sheet

| 열 이름 | 설명 |
| --- | --- |
| Substance Name | 위 시트와 연결되는 화합물 이름 |
| Source Type | 데이터 출처 유형 (예: 저널, 특허) |
| Document Identifier | DOI 또는 특허 번호 등 출처 ID |
| SMILES | 위 시트와 연결 가능 (Substance Name이 없을 경우) |
| Assay Name | 실험 이름 |
| Measurement Function | 억제제(inhibitor), 활성제(activator) 등 |
| Assay Parameter | 측정한 항목 (IC50, EC50 등) |
| Display Measurement | 단위 포함 원시 측정값 |
| Measurement Prefix (Parsed) | =, <, > 등 기호 |
| Single Value (Parsed) | 정제된 단일 측정값 (수치만) |
| Low-End / High-End Value | 측정 범위가 있을 경우의 하한/상한 |
| Measurement Unit (Parsed) | 단위 (µM, nM 등) |
| pX Value | -log(IC50), 즉 pIC50 등 |
| Cell Line | 실험에 사용된 세포주 |
| Experimental Environment | in vitro / in vivo 여부 |
| Species | 동물 종 |
| Diseases | 적응 질환 또는 표적 질환 |

### ChEMBL_ASK1(IC50).csv

| 컬럼 이름 | 의미 | 예시 값 (CHEMBL3927617 기준) |
| --- | --- | --- |
| Molecule ChEMBL ID | 화합물 고유 ID | CHEMBL3927617 |
| Molecule Name | 화합물 이름 (없을 수도 있음) | "" |
| Molecule Max Phase | 임상시험 최고 단계 (0=preclinical) | "None" |
| Molecular Weight | 분자량 (g/mol) | 359.81 |
| #RO5 Violations | 리피스키 규칙 위반 수 | 0 |
| AlogP | 소수성 예측 값 (LogP) | 2.68 |
| Compound Key | 내부 고유 키 (출처 식별자) | BDBM128413 |
| Smiles | 분자 구조 (SMILES 형식) | Cn1cc(Cl)... |
| Standard Type | 측정한 값의 종류 (IC50 등) | IC50 |
| Standard Relation | 관계 연산자 (예: =, <, >) | =' |
| Standard Value | IC50 수치 값 | 38 |
| Standard Units | 단위 (보통 nM) | nM |
| pChEMBL Value | 변환된 log 값 (-log10(IC50 in M)) | 7.42 |
| Data Validity Comment | 유효성에 대한 주석 (없을 수 있음) | "" |
| Comment | 실험 관련 코멘트 | 259344 |
| Uo Units | 단위 URI (Ontology 용도) | UO_0000065 |
| Ligand Efficiency BEI | Binding Efficiency Index | 20.62 |
| Ligand Efficiency LE | Ligand Efficiency | 0.41 |
| Ligand Efficiency LLE | Lipophilic Ligand Efficiency | 4.74 |
| Ligand Efficiency SEI | Surface Efficiency Index | 8.49 |
| Potential Duplicate | 중복 가능성 여부 (0=없음) | 0 |
| Assay ChEMBL ID | 실험(assay)의 고유 ID | CHEMBL3705948 |
| Assay Description | 실험 설명 (자세한 프로토콜) | "Homogeneous Time-Resolved Fluorescence Assay: The inhibitory..." |
| Assay Type | B=Binding, F=Functional 등 | B |
| BAO Format ID | 실험 방식 ID (BioAssay Ontology) | BAO_0000357 |
| BAO Label | 실험 방식 설명 | single protein format |
| Assay Organism | 실험 대상 유기체 | None |
| Assay Tissue ChEMBL ID | 조직 ID | None |
| Assay Tissue Name | 조직 이름 | None |
| Assay Cell Type | 세포 타입 | None |
| Assay Subcellular Fraction | 세포 내 위치 | None |
| Assay Parameters | 기타 실험 조건 | "" |
| Assay Variant Accession | 유전자 접근 번호 (있을 경우) | "" |
| Assay Variant Mutation | 변이 정보 | "" |
| Target ChEMBL ID | 타겟 단백질 ID | CHEMBL5285 |
| Target Name | 타겟 단백질 이름 | Mitogen-activated protein kinase kinase kinase 5 |
| Target Organism | 타겟 생물종 | Homo sapiens |
| Target Type | 단백질 종류 | SINGLE PROTEIN |
| Document ChEMBL ID | 문서 ID | CHEMBL3639127 |
| Source ID | 출처의 내부 ID | 37 |
| Source Description | 출처 설명 | BindingDB Patent Bioactivity Data |
| Document Journal | 발표 저널 | "" |
| Document Year | 발표 연도 | 2014 |
| Cell ChEMBL ID | 사용된 세포 ID | None |
| Properties | 기타 속성 정보 | "" |
| Action Type | 작용 방식 (예: Inhibitor) | "" |
| Standard Text Value | 수치 외 문자 값 (있을 경우) | "" |
| Value | IC50 값 중복 기입 (Standard Value와 같음) | 38 |

### PubChem_ASK1.csv

| 컬럼명 | 설명 | 예시 |
| --- | --- | --- |
| Bioactivity_ID | 각 생물활성 데이터 항목의 고유 식별자(ID) | 363645869 |
| Activity_Value | 측정된 활성이 나타나는 값 (보통 IC50, EC50 등) | 0.0001 |
| BioAssay_AID | 생물검정 실험의 고유 ID (Assay ID) | 1404085 |
| Substance_SID | 물질 식별자 (PubChem 내 서브스턴스 고유 번호) | 404713175 |
| Compound_CID | 화합물 식별자 (PubChem 내 화합물 고유 번호) | 145990765 |
| refsid | 참조 식별자 (빈 값일 수 있음) | (빈 값) |
| Gene_ID | 타겟 유전자 ID (예: NCBI Gene ID) | 4217 |
| PMID | 관련 논문 PubMed ID | 29348070 |
| Aid_Type | Assay 유형 (예: Confirmatory - 확인 시험) | Confirmatory |
| Last_Modified_Date | 데이터가 마지막으로 수정된 날짜 (YYYYMMDD) | 20220830 |
| Has_Dose_Response_Curve | 용량-반응 곡선 데이터 존재 여부 (0: 없음, 1: 있음) | 0 |
| RNAi_BioAssay | RNA 간섭 생물검정 여부 (0: 아니오, 1: 예) | 0 |
| Activity | 활성 상태 (예: Active, Inactive) | Active |
| Protein_Accession | 단백질 엑세션 번호 (예: UniProt ID) | Q99683 |
| Activity_Type | 활성 측정 타입 (예: IC50, EC50 등) | IC50 |
| Activity_Qualifier | 활성 값과 관련된 비교 기호 (예: =, >, <) | #ERROR! |
| Bioassay_Data_Source | 데이터 출처 (예: ChEMBL, PubChem) | ChEMBL |
| BioAssay_Name | 생물검정 이름 및 설명 | Inhibition of recombinant full length human GST-tagged ASK1 expressed in baculovirus ... |
| Compound_Name | 화합물 이름 | 2-methoxy-4-methyl-N-[6-(4-propan-2-yl-1,2,4-triazol-3-yl)pyridin-2-yl]-5-sulfamoylbenzamide |
| Target_Name | 타겟 단백질 이름 및 설명 | MAP3K5 - mitogen-activated protein kinase kinase kinase 5 (human) |
| Target_Link | 타겟 유전자 링크 (웹에서 클릭 가능한 경로) | /gene/4217 |
| ECs | 효소 분류번호 (EC Number) | 2.7.11.25 |
| Representative_Protein_Accession | 대표 단백질 엑세션 번호 (보통 단일 UniProt ID) | Q99683 |
| Taxonomy_ID | 생물 분류 ID (예: 9606은 인간) | 9606 |
| Cell_ID | 세포주 ID (있을 경우) | (빈 값) |
| Target_Taxonomy_ID | 타겟 생물 분류 ID | (빈 값) |
| Anatomy_ID | 해부학 ID (있으면) | (빈 값) |
| Anatomy | 해부학 명칭 (있으면) | (빈 값) |
| dois | 관련 논문 DOI | 10.1016/j.ejmech.2017.12.041 |
| pmcids | PubMed Central ID | (빈 값) |
| pclids | PMC 라이선스 ID (있으면) | (빈 값) |
| citations | 참고 문헌 인용 데이터 (문헌 리스트) | "Lovering F, Morgan P, Allais C, Aulabaugh A, ..." |
| SMILES | 화학 구조를 나타내는 SMILES 문자열 | CC1=CC(=C(C=C1S(=O)(=O)N)C(=O)NC2=CC=CC(=N2)C3=NN=CN3C(C)C)OC |