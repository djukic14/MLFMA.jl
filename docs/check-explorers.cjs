const assert = require("node:assert/strict");
const fs = require("node:fs");
const e = require("./src/assets/explorers.js");
const fixtures = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let checks = 0;
function close(a,b,tolerance=2e-11) {
    assert.equal(a.length,b.length);
    a.forEach((v,i) => { assert.ok(Math.abs(v-b[i]) <= tolerance, `${i}: ${v} != ${b[i]}`); checks++; });
}
for(const f of fixtures.sampling) {
    const s=e.sampling(f.width,f.digits);
    assert.equal(s.L,f.L); assert.equal(s.nt,f.nt); assert.equal(s.np,f.np);
    close(s.theta,f.theta);
}
for(const f of fixtures.transfers) for(const family of ["Fourier","Local Lagrange"])
    close(e.transfer(f.n,f.target,f.mode,f.stencil,family).result,f.values[family]);
for(const f of fixtures.harmonics) {
    const h=e.harmonic(f.z,f.P);
    close(h.coefficients.map(v=>v[0]),f.real);
    close(h.coefficients.map(v=>v[1]),f.imag);
}
for(const f of fixtures.phases) {
    const values=e.phase(f.delta,f.phi);
    close(values.map(v=>v[0]),f.real);
    close(values.map(v=>v[1]),f.imag);
}
assert.throws(()=>e.transfer(15,32,2,4,"Fourier"));
assert.throws(()=>e.transfer(16,32,2,20,"Local Lagrange"));
assert.throws(()=>e.harmonic(2,-1));
console.log(`${checks} numeric browser/Julia comparisons passed; invalid inputs rejected.`);
