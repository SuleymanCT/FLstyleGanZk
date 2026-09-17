# Faz E Altyapı Notu — Anvil Bellek Birikimi ve Segment Mimarisi

**Durum: kod yazıldı, Colab'da HENÜZ doğrulanmadı** (bkz.
`docs/phase_e_timing_investigation.md` — zaman araştırması bu sorundan
AYRI ve KAPANDI; ezkl'nin kendisi darboğaz DEĞİL).

## Gerçek gözlenen hata

60 ispatlık `full` mod koşusunda round 0-8 arası **36 ispat GEÇTİ**
(hepsi `verified=True`), round 9'da:

```
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1',
port=54971): Read timed out. (read timeout=30.0)
```

Round 9'un dört site'ı da ve ana döngü bu yüzden çöktü.

## Teşhis

anvil TÜM zincir durumunu bellekte tutuyor. 36 ispat boyunca 36 × ~15 KB
verifier deploy'u + 36 × ~96 KB ispat calldata'sı birikti; Colab'da her
ispat zaten ~7 GB tepe RAM yapıyor (ezkl'nin kendi bellek kullanımı,
anvil'den bağımsız). anvil şişip RPC isteklerine yanıt veremez hale
geldi.

## Uygulanan düzeltmeler

### 1. Segment mimarisi — anvil periyodik olarak TAMAMEN yeniden başlatılıyor

`scripts/replay_proofs.py`'nin ana döngüsü artık bir "segment" kavramı
etrafında kurulu: her segment KENDİ `AnvilProcess`'i + KENDİ deploy
edilmiş `RoundManager`'ı ile çalışır. `configs/schedule.yaml: replay_infra.anvil_restart_interval`
(varsayılan **10** — 36'da sorun çıktığından ~3,6× marjla çok daha
erken) kadar GERÇEK ispat biriktiğinde, bir SONRAKİ round'a geçmeden
ÖNCE mevcut segment kapatılıp yeni bir segment açılıyor:
- Yeni `AnvilProcess` başlatılıyor.
- `RoundManager.sol` YENİDEN deploy ediliyor.
- Tüm site'lar YENİDEN kaydediliyor.

Alternatif olarak değerlendirilen "her N ispatta anvil'i yeniden
başlat, ama round'un ORTASINDA da olabilir" yaklaşımı BİLEREK
seçilmedi — round sınırında (yeni round başlamadan önce) yeniden
başlatmak, o round'un TÜM site'larının AYNI `challengeSeed`'i kullanma
invaryantını round bittiği/başlamadığı sürece koruyor. (İstisna: madde
2'deki reaktif sağlık kontrolü round'un ORTASINDA da tetiklenebilir —
bu durum AYRICA ele alınıyor, aşağıya bakın.)

### 2. Her ispattan/round'dan önce sağlık kontrolü (`eth_blockNumber`)

`check_rpc_alive(rpc_url, timeout)` — `requests` ile doğrudan bir
`eth_blockNumber` JSON-RPC çağrısı, KISA bir zaman aşımıyla (varsayılan
5s, `replay_infra.health_check_timeout_seconds`). Bu, asıl işlem
`timeout`'undan (dakikalar sürebilir) BİLEREK AYRI — amaç HIZLI bir
"hayatta mı" sorgusu.

- **Round başlamadan ÖNCE**: sağlıksızsa segment yeniden başlatılır,
  `startRound` yeni segmentte çağrılır (round için sorun yok, henüz
  hiçbir site işlenmedi).
- **Her site'ın ispatından ÖNCE** (kullanıcının "her ispattan önce"
  isteği): sağlıksızsa segment yeniden başlatılır VE `startRound` O
  ROUND İÇİN TEKRAR çağrılır (YENİ bir `challengeSeed` almak için) —
  bu, o round'un ÖNCEKİ site'larının FARKLI bir `challengeSeed` ile
  ispatlanmış olabileceği anlamına gelir. **Bu bilinen bir tutarsızlık,
  GİZLENMİYOR** — `infra_events`'e (`type="unhealthy_before_proof"`)
  kaydedilip hem konsola hem `docs/phase_e_replay.md`'ye yazılıyor.

### 3. Ana döngüde anvil hatası artık ÖLÜMCÜL değil

`startRound` iki denemede de (aralarında bir segment yeniden başlatmayla)
başarısız olursa, o round'un TÜM site'ları `status="failed"` (net bir
`error_summary`'yle) işaretlenip bir SONRAKİ round'a geçiliyor — tüm
koşu düşmüyor.

### 4. `--force` olmadan resume — round düzeyinde de hızlandırıldı

Mevcut davranış (her `(round,site)` için `all_results`'ta zaten varsa
atla) DEĞİŞMEDİ ve doğru çalışıyordu. EKLENEN: `round_fully_done(round_id,
sites, all_results, force)` — bir round'un TÜM site'ları için zaten
sonuç varsa `startRound` çağrısı BİLE atlanıyor (resume sırasında
gereksiz gas/zaman harcamamak için; "full" modda anlamlı bir hızlanma,
"staged" modda genelde tetiklenmez ama YANLIŞ bir atlamaya da yol AÇMAZ).

### 5. `anvil --prune-history`

Foundry'nin belgelenmiş `--prune-history [N]` bayrağı (`AnvilProcess`'in
varsayılanı: 100 durum) eklendi — ama TEK BAŞINA yeterli olmayabileceği
BİLİNİYOR (foundry-rs/foundry#6017, #3478 — uzun koşularda bellek yine
de büyüyebiliyor). Bu yüzden madde 1'deki TAM yeniden başlatma BİRİNCİL
savunma, `--prune-history` İKİNCİL/ek bir önlem.

### 6. RPC timeout 30s → 120s

`chain/client.py: Web3Client`/`RoundManagerClient` artık varsayılan
120s (`DEFAULT_RPC_TIMEOUT_SECONDS`) kullanıyor — gözlenen hatanın
literal kaynağı (`read timeout=30.0`) buydu. `scripts/replay_proofs.py`
bunu `configs/schedule.yaml: replay_infra.rpc_timeout_seconds`'tan
okuyup açıkça geçiyor.

## Sonuç JSON'undaki yeni alanlar (makalenin ana maliyet tablosu için)

- Her `(round,site)` sonucunda `"segment_index"`: hangi anvil oturumunda
  üretildiğini gösterir.
- `{zk_root}/replay/replay_infra_events_<mod>.json`: TÜM yeniden
  başlatma/sağlık-kontrolü olaylarının tam kaydı + `restart_count` —
  raporda "altyapı kısıtı" olarak AÇIKÇA gösteriliyor, gizlenmiyor.
- `docs/phase_e_replay.md` (script tarafından üretilir) artık bir
  "altyapı olayları" bölümü de içeriyor (`render_infra_events`).

## Test kapsamı

Saf mantık (`should_restart_segment`, `round_fully_done`,
`get_replay_infra_config`, `check_rpc_alive` — gerçek ama erişilemeyen
bir port'a karşı, mock DEĞİL — `render_infra_events`) yerelde
GERÇEKTEN test edildi. Segment açma/kapama, gerçek anvil restart'ı,
gerçek `startRound` yeniden deneme mantığı BİLİNÇLİ TEST SINIRI içinde
— sadece Colab'da, gerçek bir 60 ispatlık koşumla doğrulanabilir.
