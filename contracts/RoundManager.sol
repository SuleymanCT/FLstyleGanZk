// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice ezkl'nin `create_evm_verifier` ile ürettiği Halo2Verifier'ın
/// arayüzü. Bu imza (`verifyProof(bytes,uint256[]) returns (bool)`)
/// ARTIK VARSAYIM DEĞİL — Faz D'nin gerçek Colab koşumunda GERÇEK bir
/// ezkl Verifier.sol'una (14.921 byte deployed bytecode) karşı
/// `tests/test_contracts.py` ile DOĞRULANDI: geçerli bir ispat kabul
/// edildi, tek bit bozulmuş bir ispat reddedildi — `try/catch`'in
/// "güvenli başarısızlık" dalına HİÇ düşülmeden, gerçek `bool` dönüş
/// değeri üzerinden. `try/catch` yine de KALIYOR (savunma katmanı) —
/// ezkl farklı bir devre/sürümde bu imzadan sapan bir Verifier üretirse
/// (ör. `view`/`pure` değişirse ya da dönüş tipi kaldırılırsa) çağrı
/// revert eder, `catch` bunu yakalayıp `verified=false` yapar; yani en
/// kötü ihtimalle ispat "geçersiz" sayılır (GÜVENLİ başarısızlık modu),
/// asla yanlış pozitif üretmez.
interface IHalo2Verifier {
    function verifyProof(bytes calldata proof, uint256[] calldata publicInputs) external returns (bool);
}

/// @notice Federe StyleGAN-XL turlarının zincir üstü koordinasyonu.
///
/// Faz C3'ün EIP-170 bulgusu gereği (k=4/k=8 verifier'ları zincire deploy
/// edilemiyor, bkz. docs/phase_c_report.md) protokol k=1 ile çalışır: her
/// `submitProof` çağrısı TEK bir (z,c) meydan okumasını ispatlar.
///
/// **ÖNEMLİ TASARIM NOTU — tek bir sabit `verifier` YOK, `submitProof`
/// verifier adresini PARAMETRE olarak alır.** Faz B/C boyunca ezkl
/// `param_visibility="fixed"` kullanıldı (ağırlıklar devrenin SABİT
/// sütunları — `vk`/verifier bytecode'unun İÇİNE gömülü). Federe öğrenmede
/// her round/site'ın ağırlıkları FARKLI olduğundan, HER (round,site)
/// ispatının KENDİ verifier kontratı olur — tek bir sabit `verifier`
/// adresi round 2'den itibaren yanlış olurdu. `round_runner.py` bu yüzden
/// her gerekli ispat için TAZE bir Verifier deploy edip adresini
/// `submitProof`'a veriyor; kontrat bunu `Submission.verifierUsed`'a
/// kaydedip sonradan denetlenebilir kılıyor.
///
/// Ağırlık verisi HİÇBİR ZAMAN zincire yazılmaz — sadece CID (IPFS),
/// hash/taahhüt (bytes32) ve ispat (bytes) tutulur.
contract RoundManager {
    address public immutable owner;

    uint256 public immutable reputationInitial;
    uint256 public immutable reputationPenalty;
    uint256 public immutable reputationBonus;
    uint256 public immutable reputationThreshold;

    struct RoundInfo {
        string globalCID;
        bytes32 globalHash;
        bytes32 challengeSeed;
        uint256 startedAt;
        bool finalized;
        string aggregateCID;
        bool exists;
    }

    struct Submission {
        string updateCID;
        bytes32 weightCommitment;
        uint256 submittedAt;
        bool proofSubmitted;
        bool proofVerified;
        address verifierUsed;
    }

    mapping(address => bool) public registeredSites;
    mapping(address => int256) public reputation;

    mapping(uint256 => RoundInfo) private rounds;
    mapping(uint256 => mapping(address => Submission)) private submissions;

    event SiteRegistered(address indexed site);
    event RoundStarted(uint256 indexed roundId, string globalCID, bytes32 globalHash, bytes32 challengeSeed);
    event UpdateSubmitted(uint256 indexed roundId, address indexed site, string updateCID, bytes32 weightCommitment);
    event ProofSubmitted(uint256 indexed roundId, address indexed site, bool verified, int256 newReputation);
    event SiteExcluded(uint256 indexed roundId, address indexed site, int256 reputation);
    event RoundFinalized(uint256 indexed roundId, string aggregateCID, address[] includedSites);

    modifier onlyOwner() {
        require(msg.sender == owner, "RoundManager: sadece owner");
        _;
    }

    modifier onlyRegisteredSite() {
        require(registeredSites[msg.sender], "RoundManager: kayitli site degil");
        _;
    }

    constructor(
        uint256 _reputationInitial,
        uint256 _reputationPenalty,
        uint256 _reputationBonus,
        uint256 _reputationThreshold
    ) {
        owner = msg.sender;
        reputationInitial = _reputationInitial;
        reputationPenalty = _reputationPenalty;
        reputationBonus = _reputationBonus;
        reputationThreshold = _reputationThreshold;
    }

    function registerSite(address site) external onlyOwner {
        require(site != address(0), "RoundManager: site adresi sifir olamaz");
        require(!registeredSites[site], "RoundManager: zaten kayitli");
        registeredSites[site] = true;
        reputation[site] = int256(reputationInitial);
        emit SiteRegistered(site);
    }

    /// @notice `challengeSeed`'i `blockhash(block.number - 1)` ve `roundId`'den
    /// türetir — site tarafı bunu (`getChallengeSeed`) okuyup deterministik
    /// (z, c) üretir (bkz. orchestrator/challenge.py).
    function startRound(uint256 roundId, string calldata globalCID, bytes32 globalHash) external onlyOwner {
        require(!rounds[roundId].exists, "RoundManager: round zaten baslatildi");
        bytes32 challengeSeed = keccak256(abi.encodePacked(blockhash(block.number - 1), roundId));
        rounds[roundId] = RoundInfo({
            globalCID: globalCID,
            globalHash: globalHash,
            challengeSeed: challengeSeed,
            startedAt: block.timestamp,
            finalized: false,
            aggregateCID: "",
            exists: true
        });
        emit RoundStarted(roundId, globalCID, globalHash, challengeSeed);
    }

    function submitUpdate(uint256 roundId, string calldata updateCID, bytes32 weightCommitment) external onlyRegisteredSite {
        RoundInfo storage r = rounds[roundId];
        require(r.exists, "RoundManager: round yok");
        require(!r.finalized, "RoundManager: round zaten kapandi");
        require(submissions[roundId][msg.sender].submittedAt == 0, "RoundManager: bu round'a zaten gonderim yapildi");

        submissions[roundId][msg.sender] = Submission({
            updateCID: updateCID,
            weightCommitment: weightCommitment,
            submittedAt: block.timestamp,
            proofSubmitted: false,
            proofVerified: false,
            verifierUsed: address(0)
        });
        emit UpdateSubmitted(roundId, msg.sender, updateCID, weightCommitment);
    }

    /// @notice `verifierAddress.verifyProof`'u çağırır (bkz. yukarıdaki
    /// "ÖNEMLİ TASARIM NOTU" — verifier her (round,site) için FARKLI
    /// olabilir, o yüzden parametre olarak alınır). Geçerse itibar
    /// `+reputationBonus`, geçmezse `-reputationPenalty`; eşik altına
    /// düşerse `SiteExcluded` yayılır (dışlama `finalizeRound`/`isSiteEligible`
    /// tarafında UYGULANIR, burada sadece SİNYALLENİR).
    function submitProof(uint256 roundId, address verifierAddress, bytes calldata proof, uint256[] calldata publicInputs)
        external
        onlyRegisteredSite
    {
        require(verifierAddress != address(0), "RoundManager: verifier adresi sifir olamaz");
        RoundInfo storage r = rounds[roundId];
        require(r.exists, "RoundManager: round yok");
        require(!r.finalized, "RoundManager: round zaten kapandi");
        Submission storage s = submissions[roundId][msg.sender];
        require(s.submittedAt != 0, "RoundManager: once submitUpdate gerekli");
        require(!s.proofSubmitted, "RoundManager: bu round icin ispat zaten gonderildi");

        bool verified;
        try IHalo2Verifier(verifierAddress).verifyProof(proof, publicInputs) returns (bool ok) {
            verified = ok;
        } catch {
            verified = false;
        }

        s.proofSubmitted = true;
        s.proofVerified = verified;
        s.verifierUsed = verifierAddress;

        int256 newReputation = verified
            ? reputation[msg.sender] + int256(reputationBonus)
            : reputation[msg.sender] - int256(reputationPenalty);
        reputation[msg.sender] = newReputation;

        emit ProofSubmitted(roundId, msg.sender, verified, newReputation);

        if (newReputation < int256(reputationThreshold)) {
            emit SiteExcluded(roundId, msg.sender, newReputation);
        }
    }

    /// @notice Sadece DOĞRULANMIŞ (`proofVerified=true`) VEYA ispat HİÇ
    /// İSTENMEMİŞ (`proofSubmitted=false` — bu round'da takvim gereği
    /// ispat beklenmiyordu) siteler dahil edilebilir. İspat gönderilip
    /// BAŞARISIZ olan site kesin REDDEDİLİR.
    function finalizeRound(uint256 roundId, string calldata aggregateCID, address[] calldata includedSites) external onlyOwner {
        RoundInfo storage r = rounds[roundId];
        require(r.exists, "RoundManager: round yok");
        require(!r.finalized, "RoundManager: round zaten kapandi");

        for (uint256 i = 0; i < includedSites.length; i++) {
            address site = includedSites[i];
            require(registeredSites[site], "RoundManager: kayitsiz site dahil edilemez");
            Submission storage s = submissions[roundId][site];
            bool eligible = !s.proofSubmitted || s.proofVerified;
            require(eligible, "RoundManager: dogrulanmamis site finalizeRound'a dahil edilemez");
        }

        r.finalized = true;
        r.aggregateCID = aggregateCID;
        emit RoundFinalized(roundId, aggregateCID, includedSites);
    }

    // --- Görünüm fonksiyonları ---

    function getChallengeSeed(uint256 roundId) external view returns (bytes32) {
        require(rounds[roundId].exists, "RoundManager: round yok");
        return rounds[roundId].challengeSeed;
    }

    function isSiteEligible(address site) external view returns (bool) {
        return registeredSites[site] && reputation[site] >= int256(reputationThreshold);
    }

    function getRoundInfo(uint256 roundId)
        external
        view
        returns (
            string memory globalCID,
            bytes32 globalHash,
            bytes32 challengeSeed,
            uint256 startedAt,
            bool finalized,
            string memory aggregateCID
        )
    {
        RoundInfo storage r = rounds[roundId];
        require(r.exists, "RoundManager: round yok");
        return (r.globalCID, r.globalHash, r.challengeSeed, r.startedAt, r.finalized, r.aggregateCID);
    }

    /// @notice `getChallengeSeed`/`isSiteEligible`/`getRoundInfo` dışında,
    /// bir (round,site) gönderiminin tam durumunu okumak için gereken EK
    /// bir görünüm fonksiyonu (spesifikasyonda açıkça istenmedi, ama
    /// `submitProof` sonrası `proofVerified`'ı zincirden geri okumak için
    /// gerekli — round_runner.py/tests/test_contracts.py bunu kullanır).
    function getSubmission(uint256 roundId, address site)
        external
        view
        returns (
            string memory updateCID,
            bytes32 weightCommitment,
            uint256 submittedAt,
            bool proofSubmitted,
            bool proofVerified,
            address verifierUsed
        )
    {
        Submission storage s = submissions[roundId][site];
        return (s.updateCID, s.weightCommitment, s.submittedAt, s.proofSubmitted, s.proofVerified, s.verifierUsed);
    }
}
