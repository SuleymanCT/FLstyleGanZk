"""Faz F tespit analizi — hangi mekanizma bir saldırıyı YAKALAR, hangisi
KAÇIRIR.

**ZK'nın gerçekte ne kanıtladığının koda dayalı analizi (önemli
düzeltme):** `orchestrator/round_runner.py: generate_and_submit_proof`
incelendiğinde görülür ki weight commitment (`compute_weight_commitment`)
ve ZK devresi HER ZAMAN AYNI `mapping_shard`/`full_state` objesinden
türetilir (`run_round`'da tek bir `full_state` hem `weight_commitment`
hem `mapping_shard` hem `site_states[site]` için kullanılıyor) — bu kod
tabanında bir sitenin "taahhüt ettiği ağırlıkla FARKLI bir ağırlığı
ispatlayıp/katkıda bulunması" için HİÇBİR kod yolu yok. Yani ZK, GERÇEK
ANLAMDA, gönderilen ağırlığın İÇERİĞİNİ (iyi/kötü/zehirli) DEĞİL,
taahhüt→ispat→katkı ÜÇLÜSÜNÜN TUTARLILIĞINI kanıtlıyor. Faz F'in
simüle ettiği üç saldırı (random_weights/scaled_poison/conditional_poison)
DÜRÜST bir istemcinin KENDİ zehirli ağırlığını taahhüt edip ispatlayıp
katkıda bulunduğu senaryolar — hiçbiri bir taahhüt/ispat UYUŞMAZLIĞI
içermiyor, bu yüzden ZK YAPISAL OLARAK hiçbirini yakalayamaz. Bu
VARSAYILMIYOR, `verify_commitment_consistency` ile GERÇEKTEN
hesaplanıyor (üç saldırı için de sonucun `True` çıkması beklenir —
ama bu beklenti kodda hardcode edilmiyor, her seferinde yeniden
hesaplanıyor).

Norm kontrolü BURADA yeniden yazılmıyor — `orchestrator/aggregate.py:
compute_delta_norm`/`gate_site_update` zaten mevcut, `scripts/run_attacks.py`
onu doğrudan çağırır.
"""

from __future__ import annotations

from storage.hashing import canonical_hash

# ZK devresi SADECE mapping alt-ağını kapsıyor (`param_visibility="fixed"`,
# bkz. `contracts/RoundManager.sol`'un tasarım notu ve
# `orchestrator/round_runner.py: MAPPING_PREFIXES=["mapping"]`) —
# synthesis ağı (parametrelerin BÜYÜK çoğunluğu) HİÇBİR ZK devresine
# GİRMEZ, ZK o kısım hakkında hiçbir şey ispatlamaz/kanıtlamaz.
ZK_PROVEN_PREFIX = "mapping"


def verify_commitment_consistency(committed_state: dict, contributed_state: dict) -> bool:
    """`committed_state` (zincire `weightCommitment` olarak taahhüt
    edilen ağırlık kümesi) ile `contributed_state` (FedAvg'a FİİLEN
    katkıda bulunan ağırlık kümesi) AYNI mı — `storage.hashing.canonical_hash`
    eşitliğiyle GERÇEKTEN kontrol edilir (varsayılmaz).

    `True`  -> tutarlı: ZK/commitment mekanizmasının YAKALAYACAĞI bir
               TUTARSIZLIK YOK (yakalayabileceği tek şey budur).
    `False` -> bait-and-switch: taahhüt edilenle katkıda bulunulan
               FARKLI — ZK/commitment BUNU yakalardı. Bu projedeki
               dürüst-istemci akışında (`generate_and_submit_proof`)
               bu durum OLUŞMAZ; sadece gerçek bir kötü niyetli
               istemci implementasyonu (kod tabanı DIŞI) üretebilirdi."""
    return canonical_hash(committed_state) == canonical_hash(contributed_state)


def attack_touches_zk_proven_scope(modified_keys) -> bool:
    """Değiştirilen anahtarlardan en az biri `mapping.` ile başlıyorsa
    `True` — saldırı ZK'nın ispatladığı alt-ağın (mapping) İÇİNDE.
    Bu, "kapsam içi mi" sorusuna cevap verir — "kapsam içi olsa bile
    YAKALANIR mı" sorusuna DEĞİL (o soruyu `verify_commitment_consistency`
    cevaplıyor; bu projenin dürüst-istemci akışında cevap HER ZAMAN
    hayırdır, kapsam içi/dışı fark etmeksizin)."""
    return any(str(key).startswith(f"{ZK_PROVEN_PREFIX}.") for key in modified_keys)
