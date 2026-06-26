import stt_services, json, os, glob

samples_dir = 'samples/01_accuracy_benchmarks'
samples = sorted(glob.glob(samples_dir + '/sample_*.wav'))
results = []

for s in samples:
    name = os.path.basename(s)
    try:
        r = stt_services.transcribe_auto(s)
        lang   = r['telemetry']['detected_language']
        engine = r['telemetry']['engine_used']
        text   = r['transcript'][:120]
        results.append({'file': name, 'lang': lang, 'engine': engine, 'text': text})
        print(name + ': [' + lang + '] via ' + engine)
        print('  >> ' + r['transcript'][:100])
    except Exception as e:
        results.append({'file': name, 'error': str(e)})
        print(name + ': ERROR - ' + str(e))

with open('sample_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('Done - saved sample_results.json')
