# Snug

Local Edge AI thermal home check for Junction 2026 (Arduino Hardware / Edge AI challenge).  
A Topdon TC001 on an Arduino UNO Q running a web UI guides you through your house to look at the outer walls, **on device**, six questions:

1. What building element is this?  
2. Is the thermal pattern normal?  
3. Insulation gap, thermal bridge, or air leak?  
4. How severe?  
5. Which fix should be first?  
6. How do you verify the fix? (re-scan; ΔT and anomaly area should drop)

Each defect is also labelled with how much you **save per year** if you fix it (Finnish climate estimate), and the dashboard shows a **kit cost** card: Topdon TC001, Arduino UNO Q, and Snug (open source) at €0, plus a payback time.

Nothing is sent to the cloud. A language-model explanation helper is **not** included in the default path.

<img width="1280" height="686" alt="snug screenshot" src="https://github.com/user-attachments/assets/94fbbae9-7d08-477d-879f-4557a4abe818" />

---

## Svenska — körning

### På datorn (utan kamera)

```bash
cd snug
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_detector.py          # redan tränad modell ligger i models/
python server.py --simulate
# öppna http://127.0.0.1:8080
# macOS + TC001: ./run_mac.sh   (imageio-ffmpeg installeras från requirements.txt)
```

Tester och benchmark:

```bash
pip install -r requirements-dev.txt   # eller bara pytest
python -m pytest
python scripts/benchmark.py
```

### På Arduino UNO Q (Debian aarch64)

**Hårdvara (verifierat mot Arduino-dokumentation):** Qualcomm Dragonwing QRB2210, 4× Cortex-A53 @ 2.0 GHz, 2 GB eller 4 GB LPDDR4, Debian Linux, plus STM32U585 (Cortex-M33). En USB-C-port. Kameran och matning behöver en **USB-C PD-hubb/docka** så att kortet får ström samtidigt som TC001 sitter i host-läget.

```bash
# På UNO Q, efter att du kopierat den här mappen:
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Bekräfta kameran:
lsusb | grep -i 0bda || lsusb
python -m snug.capture --frames 10          # eller --simulate om kameran saknas
python server.py --host 0.0.0.0 --port 8080
# Öppna http://<uno-q-ip>:8080 från din dator
```

Om `onnxruntime` saknar en aarch64-wheel räcker `models/detector.npz` — inferensen körs då i NumPy och kräver inte ONNX.

**Arduino App Lab (valfritt larm):** Python på Linux-sidan pratar med MCU-skissen via den officiella Bridge-API:n (`arduino.app_utils.Bridge`, `arduino-router`). Öppna **inte** `/dev/ttyHS1` eller `Serial1` själv. Ladda `sketches/AlarmBridge/AlarmBridge.ino` till STM32 och starta dashboarden med `--bridge`. Då anropas `Bridge.call("set_alarm", level, timeout=2)` (0–3) när allvarligheten ändras. Dokument: <https://docs.arduino.cc/software/app-lab/bridge/bridge-api/>

### Egen datainsamling och finjustering

Detektorn i `models/` är tränad på **syntetiska** 256×192-scener. Den måste finjusteras på riktiga TC001-bilder av modellhuset innan ni litar på auto-rutor på scenen. För demon fungerar `detect_mode: "hybrid"` / `--rois-only` med rutorna i `config.example.json`.

```bash
python scripts/collect_images.py --device 0 --count 200
# PNG-filerna laddas upp till Roboflow (bbox: window, door, wall, roof, corner)
# eller Edge Impulse (FOMO). Behåll .npy-filerna — de är de radiometriska matriserna.
# Exportera YOLO/COCO och träna om med en modifierad train_detector.py, eller träna FOMO i EI
# och byt ut anropet i snug/detect.py.
```

Reglerna för defekttyp, allvarlighet, Q = U·A·ΔT och den årliga merkostnaden är **uppskattningar** (Estimate), inte en energideklaration. Metoden beskrivs nedan.

---

## English — overview

**Pipeline.** TC001 UVC frames are 256×384 YUYV; the bottom half is 16-bit radiometry, `temp_C = (lo + hi*256)/64 - 273.15`. Always set `CAP_PROP_CONVERT_RGB` to `0`. A house simulator produces the same shape with an insulated wall, uninsulated wall, window, leaky door, corners/roof, and a ramped indoor heat source. Defects (blob / linear bridge / streaky fluttering leak) are configurable.

A tiny FOMO-style CNN (1×192×256 → 6-class 12×16 grid, NumPy or ONNX Runtime) proposes element boxes. If weights are missing, or in `hybrid` mode, named ROIs from `config.example.json` are used — that is the reliable path for a live demo.

Per region, rule/statistical analysis on the **Celsius matrix** decides normality (ΔT vs the insulated reference and ambient), defect type (shape + edge overlap + short-term fluctuation), severity, a labelled heat-loss estimate `Q = U·A·ΔT`, a priority rank, and an action + verification sentence. The dashboard can store a baseline and compare a re-scan.

**UNO Q setup** is above. Use a USB-C PD hub for the camera. Optional MCU alarm via App Lab Bridge; see the Swedish section and `sketches/AlarmBridge`.

**Honesty.** The shipped detector was trained only on simulated frames (grid accuracy is modest, especially for the small door class). Fine-tune on real TC001 images. Public building-thermography sets help as extras, not as a substitute — see `/workspace/junction2026-thermal/DATASETS.md` if you have that tree.

**Commands on the real UNO Q tomorrow:**

```bash
cd snug && source .venv/bin/activate
python -m pytest
python server.py --simulate                 # no camera
python server.py --host 0.0.0.0 --port 8080 # with TC001
# optional:
python server.py --bridge --host 0.0.0.0 --port 8080
```




---

## Detection sensitivity and live (interior) surveys

`sensitivity` = `low` | `normal` | `high` (config `"sensitivity"`, CLI `--sensitivity`, or `POST /api/cost/settings {"sensitivity": "high"}`; `GET /api/state` returns `sensitivity`, `sensitivity_options`, `sensitivity_presets`).

| preset | normal ΔT | anomaly offset | min area | medium / high ΔT |
|---|---|---|---|---|
| low (original demo thresholds) | 1.5 °C | 1.2 °C | 2 % | 3.0 / 5.0 °C |
| normal | 1.0 °C | 0.8 °C | 2 % | 2.5 / 4.5 °C |
| high (live-camera default) | 0.5 °C | 0.4 °C | 1 % | 1.5 / 3.0 °C |

Each region is tested for both a warm excess (exterior view / model house) and a cold deficit (interior view: leaks and cold bridges are cold spots); the larger one wins (`direction` = `warm`|`cold`). Values are EMA-smoothed (~1 s), the mask is box-filtered and blobs under 6 px are ignored. The simulator defaults to `normal`.

With a live camera (no simulator meta) the cost model treats the scan as an interior survey: T_in = measured scene temperature (frame median) unless you set `t_in_c`, T_out = `cost.t_out_c` or `cost.default_t_out_c` (0 °C, Finnish heating season). The frame border is never used. Warm spots seen from inside are not counted as heat loss.

## Årlig merkostnad för värmeläckage / Annual extra heating cost

### Svenska — metod, antaganden, källor och begränsningar

Varje upptäckt defekt får en **uppskattning** av hur mycket man sparar per år i extra uppvärmning om felet åtgärdas, i Finlands klimat. Siffran visas som ett intervall (t.ex. *Fix → save ~€9–13/yr (Estimate)*) just för att metoden inte ska ge falsk precision. På värmebilden och i dashboarden summeras det som *Fixing these saves ~€X–Y / year*. En kit-ruta visar officiella priser: Topdon TC001 €225 exkl. moms (<https://eu.topdon.com/products/tc001>), Arduino UNO Q 2 GB €59,90 inkl. moms (<https://store.arduino.cc/products/uno-q>; 4 GB €82,90 inkl. moms, <https://store.arduino.cc/products/uno-q-4gb>), och Snugs öppen källkod €0. Priserna kontrollerades 14 november 2026 (Europe/Helsinki). Återbetalningstid = kitsumma / årlig besparing.

**Yttemperatur → effektivt U.** Värmeflödet skattas från yttemperaturen:

- inifrån (EN ISO 6946 / YM Energiatehokkuus tabell 2): `q = (T_in − T_yta) / R_si`, `R_si = 0,13 m²K/W` (horisontell värmeström; 0,10 uppåt, 0,17 nedåt)
- utifrån (samma tabell): `q = (T_yta − T_ute) / R_se`, `R_se = 0,04 m²K/W`

`U_eff = q / (T_in − T_ute)`. Tilläggsförlusten är `ΔU = U_defekt − U_referens` över defektens yta. Referensen är den isolerade/normala ytan (isolerad vägg för vägg/hörn, elementets egen normalyta för fönster och dörr). Kameran väljer automatiskt in- eller utsida beroende på om ytan ligger närmare inne- eller uteluften — modellhusdemon filmas typiskt utifrån.

**Skalmodell → verklig yta.** Defektens andel av elementets ROI multipliceras med en antagen verklig area (standard: fönster 1,5 m², dörr 2,0 m², väggsektion 10 m², tak 10 m², hörn 2,0 m²). Allt är ställbart i `config.example.json` och i dashboarden.

**Årsenergi.** `E_kWh = ΔU · A · S17 · 24 / 1000`. S17 är Ilmatieteen laitoksen lämmitystarveluku för normalperioden 1991–2020. För kommuner som inte är en av de 16 officiella jämförelseorterna (t.ex. Pietarsaari, Espoo, Rovaniemi) används närmaste jämförelsestation och det officiella K1-talet: `S_kommun = S_jämförelse / K1`. Källor: <https://www.ilmatieteenlaitos.fi/lammitystarveluvut>, CSV <https://cdn.fmi.fi/documents/climate/wau2020/lammitystarveluvut_1991-2020.csv> och <https://cdn.fmi.fi/documents/climate/wau2020/kuntakertoimet_2023.csv>.

Inlagda årsvärden (S17, °C·d, avrundat): Helsingfors 3803 (station Helsinki-Kaisaniemi, K1 0,97), Esbo 3883 (samma station, K1 0,95), Vasa 4192 (K1 1,02), Jakobstad 4363 (station Vasa, K1 0,98), Karleby 4598 (station Vasa, K1 0,93), Uleåborg 5035 (K1 0,96), Jyväskylä 4531 (K1 1,02), Rovaniemi 5642 (station Sodankylä, K1 1,05), Sodankylä 6107 (K1 0,97).

**Kostnad.** `€/år = E_kWh / verkningsgrad_eller_COP × pris`. Standardpriser från Tilastokeskus, *Energian hinnat*, juni 2026 (publicerat 24.9.2026, <https://stat.fi/fi/tilasto/ehi>):

- El, fristående pientalo: 160 €/MWh (tabell 13nl); intervall 14,892–17,91 snt/kWh för hushåll >15 000 kWh respektive 5 000–15 000 kWh (13rb).
- Direktel: COP 1. Luftvärmepump: säsongs-COP 2,5–3 (antagande). Bergvärme: ca 3–3,5 (antagande).
- Fjärrvärme, pientalo: 114 €/MWh (13nl) / 114,39 €/MWh (12gd); radhus 109,49 €/MWh.
- Lätt eldningsolja: 167,5 €/MWh (12ge; 1,67 €/l), pannverkningsgrad antagen 0,85–0,90.

Pris, ort, värmetyp och T_in/T_ute går att ändra i dashboarden, i `config.example.json` och med `--locality`, `--heating-type`, `--t-in`, `--t-out`, `--price`. Standard är Vasa och luftvärmepump. I simulatorn (modellhuset) används simulatorns inne-/utetemperatur om de inte överskrivs. Med riktig kamera (inga simulatordata) görs en *invändig* skanning: T_in är den uppmätta scentemperaturen (median av bilden) om man inte fastställt `--t-in`/`cost.t_in_c`, och T_ute är `cost.t_out_c`/`--t-out` eller annars `cost.default_t_out_c` (0 °C, typisk finsk uppvärmningssäsong). Bildens kant används aldrig som utetemperatur.

**Känslighet.** `sensitivity` = `low` | `normal` | `high` (`config.example.json`, `--sensitivity`, eller `POST /api/cost/settings {"sensitivity": "high"}`; aktuellt värde och alternativ finns i `/api/state` som `sensitivity` och `sensitivity_options`). `low` är de ursprungliga modellhus-trösklarna (normal ΔT 1,5 °C, avvikelse 1,2 °C, medium 3, hög 5). `normal` = 1,0 / 0,8 / medium 2,5 / hög 4,5 och `high` = 0,5 / 0,4 / medium 1,5 / hög 3,0 °C med minsta anomalyyta 1 %. Standard är `high` för riktig kamera och `normal` i simulatorn. Varje region testas åt båda hållen: varm avvikelse (modellhuset sett utifrån) och *kall* avvikelse (köld­brygga/luftläcka sedd inifrån); den större vinner. Resultaten jämnas ut med ett exponentiellt medelvärde (~1 s), bilden mjukas upp 3×3 och anomalier under 6 pixlar ignoreras så att sensorbrus inte ger falska fynd. En liten kall fläck (0,8–1,5 °C, 5–15 % av en vägg) ger i storleksordningen några euro per år; en liten fläck på ett fönster (1,5 m²) ger runt 1 €/år. Siffrorna är fortfarande uppskattningar.

**Begränsningar.** Yttemperaturmetoden är ungefärlig. Den kräver typiskt ΔT ≳ 10 °C inne–ute under skanningen (annars märks uppskattningen som osäker). FMI:s S17 antar innertemperatur +17 °C (interna laster täcker resten upp till vanlig rumstemperatur) — det är inte samma sak som skanningens T_in. Luftläckage skattas med samma yttemperaturmetod; extra infiltrationsförluster modelleras inte och kan underskattas. Skalmodellens areor är antaganden. Detta är ingen energideklaration eller certifierad audit.

YM-källa för R_si/R_se: <https://ym.fi/documents/1410903/0/Rakennuksen%20U-arvo_laskenta_Energiatehokkuus%2020250311%20(5).pdf>

### English — method, assumptions, sources and limitations

Each detected defect gets a labelled **Estimate** of extra annual heating cost in Finland's climate, shown as a range (for example *€40–60 / year (Vaasa climate, air-source heat pump)*) so the figure is not mistaken for a precise invoice.

Effective U is inferred from surface temperature: `q = (T_in − T_surface) / R_si` on an interior survey (`R_si = 0.13 m²K/W`, horizontal flow, EN ISO 6946 / Finnish Ministry of the Environment Energiatehokkuus Table 2) or `q = (T_surface − T_out) / R_se` from outside (`R_se = 0.04 m²K/W`). Then `U_eff = q / (T_in − T_out)` and `ΔU = U_defect − U_ref` over the defect area. Because the demo is a scale model, anomaly area is mapped with `A = (anomaly fraction of the element ROI) × assumed real size` (defaults: window 1.5 m², door 2.0 m², wall section 10 m², roof 10 m², corner 2.0 m²).

Annual energy is `E_kWh = ΔU · A · S17 · 24 / 1000`, using Finnish Meteorological Institute heating degree days (lämmitystarveluku S17, 1991–2020 normal). Municipalities that are not one of the 16 official comparison towns use the official station plus K1 coefficient (`S_local = S_ref / K1`). Sources: <https://www.ilmatieteenlaitos.fi/lammitystarveluvut>, <https://cdn.fmi.fi/documents/climate/wau2020/lammitystarveluvut_1991-2020.csv>, <https://cdn.fmi.fi/documents/climate/wau2020/kuntakertoimet_2023.csv>.

Cost is `€/year = E_kWh / efficiency_or_COP × price`. Default prices are Statistics Finland *Energian hinnat* for June 2026 (released 24 Sep 2026, <https://stat.fi/fi/tilasto/ehi>): electricity 160 €/MWh (table 13nl) with a 14.892–17.91 snt/kWh household band (13rb); district heat 114–114.39 €/MWh for a detached house and 109.49 €/MWh for a terraced/small block (13nl, 12gd); light fuel oil 167.5 €/MWh (12ge). Direct electric COP = 1; air-source seasonal COP 2.5–3 and ground-source ~3–3.5 are typical assumptions, not measured SPF. Oil uses an assumed boiler efficiency of 0.85–0.90. Locality, heating type, price and T_in/T_out are configurable (defaults: Vaasa, air-source heat pump, T_in 21 °C, T_out from the measurement or simulator).

The dashboard shows that estimate as *Fix → save ~€X–Y/yr (Estimate)* on each defect in the thermogram, with a banner *Fixing these saves ~€X–Y / year*. A kit-cost card lists official hardware prices (editable in `config.example.json`) and *Pays back in ~N months* = kit total / yearly savings. Prices checked 14 Nov 2026 (Europe/Helsinki): Topdon TC001 €225 excl. VAT from the official TOPDON Europe store (<https://eu.topdon.com/products/tc001>); Arduino UNO Q 2GB €59.90 VAT included (<https://store.arduino.cc/products/uno-q>) and 4GB €82.90 VAT included (<https://store.arduino.cc/products/uno-q-4gb>); Snug software €0 (open source). The TC001 official list is excl. VAT while the UNO Q official price is VAT included — both are shown as published. Default kit total is €284.90 using the 2GB board.

The surface-temperature method is approximate. It needs a scan ΔT of about 10 °C or more, maps a scale-model anomaly onto assumed real areas, and treats air leaks via the surface method only (infiltration is not modelled and may be underestimated). FMI S17 uses a 17 °C indoor balance temperature, which is not the same as the scan's T_in. This is not a certified energy audit. Model-house figures are demo-sized.

---

## Benchmark (build machine)

See `docs/BENCHMARK.md`. Native x86 was hundreds of FPS. The same suite under **qemu-user aarch64** (4 cores, NumPy detector, optional 2 GiB memory cap) stayed around **11 FPS**, already above the 5 FPS target. qemu-user is slower than a real UNO Q (Cortex-A53 @ 2.0 GHz). On the board, install `onnxruntime` if the aarch64 wheel imports cleanly; if not, the NumPy weights in `models/detector.npz` are enough. Set `SNUG_NO_ONNX=1` only if ONNX Runtime crashes (observed under qemu-user, not expected on real silicon).

## Tested on a real Arduino UNO Q (14 Nov 2026)

Deployed over USB with `adb`, no internet needed on the board: `./deploy/deploy_unoq.sh [SSID PASSWORD]`. It installs aarch64 Python wheels, copies Snug to `/root/snug` and enables the `snug` systemd service (`deploy/snug.service`), which waits for the TC001 and starts automatically at boot. Then connect the UNO Q to a USB-C PD hub with power + TC001 and open `http://<board-ip>:8080` from any device on the same Wi-Fi (the network must allow device-to-device traffic).

Measured on the UNO Q 2GB (simulator input): detection ~6 ms, analysis ~18 ms per frame, ~5 FPS end to end (most time is drawing/encoding the dashboard image). The live TC001 stream runs at 25 FPS on the Mac dev setup.
