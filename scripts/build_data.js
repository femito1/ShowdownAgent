// Dumps entity vocabularies and random-battle movepools from the Showdown checkout.
// Usage: node scripts/build_data.js  (after `node build` in pokemon-showdown/)
const fs = require('fs');
const path = require('path');
const {Dex, toID} = require('../pokemon-showdown/dist/sim');

const OUT = path.join(__dirname, '..', 'src', 'porygon', 'data');
const ids = table => [...new Set(table.all().map(e => e.id))].sort();

fs.writeFileSync(path.join(OUT, 'vocab.json'), JSON.stringify({
  species: ids(Dex.species),
  moves: ids(Dex.moves),
  items: ids(Dex.items),
  abilities: ids(Dex.abilities),
}));

// Per-species union of every role's movepool/abilities: a prior over unrevealed opponent sets.
for (const gen of ['gen4', 'gen9']) {
  const file = path.join(__dirname, '..', 'pokemon-showdown', 'data', 'random-battles', gen, 'sets.json');
  const sets = JSON.parse(fs.readFileSync(file, 'utf8'));
  const out = {};
  for (const [species, entry] of Object.entries(sets)) {
    const moves = new Set(), abilities = new Set();
    for (const set of entry.sets) {
      set.movepool.forEach(m => moves.add(toID(m)));
      (set.abilities || []).forEach(a => abilities.add(toID(a)));
    }
    out[species] = {level: entry.level, moves: [...moves].sort(), abilities: [...abilities].sort()};
  }
  fs.writeFileSync(path.join(OUT, `${gen}randombattle_sets.json`), JSON.stringify(out));
}
