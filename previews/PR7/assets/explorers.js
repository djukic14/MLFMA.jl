/* Small browser illustrations. docs/check-explorers.jl compares them with Julia. */
(function (root) {
    "use strict";
    const tau = 2 * Math.PI;
    const grid = (n, end = tau) => Array.from({length: n}, (_, i) => end * i / n);
    const integer = (n, min = 0) => {
        if (!Number.isSafeInteger(n) || n < min) throw Error(`Use an integer >= ${min}.`);
        return n;
    };
    function legendre(n, x) {
        let a = 1, b = x;
        if (!n) return a;
        for (let l = 2; l <= n; l++) [a, b] = [b, ((2*l-1)*x*b-(l-1)*a)/l];
        return b;
    }
    function gauss(n) {
        integer(n, 1);
        const x = Array(n), w = Array(n);
        for (let i = 0; i < Math.ceil(n / 2); i++) {
            let z = Math.cos(Math.PI*(i + 0.75)/(n + 0.5)), derivative;
            for (let j = 0; j < 50; j++) {
                const p = legendre(n, z);
                derivative = n*(z*p-legendre(n-1, z))/(z*z-1);
                const step = p / derivative;
                z -= step;
                if (Math.abs(step) < 2e-15) break;
                if (j === 49) throw Error("Quadrature did not converge at this order.");
            }
            derivative = n*(z*legendre(n,z)-legendre(n-1,z))/(z*z-1);
            x[i] = -z; x[n-1-i] = z;
            w[i] = w[n-1-i] = 2/((1-z*z)*derivative*derivative);
        }
        return {x, w};
    }
    function sampling(width, digits) {
        if (!(width > 0 && digits > 0 && Number.isFinite(width + digits))) throw Error("Use positive width and accuracy digits.");
        const kd = tau * Math.sqrt(3) * width;
        const L = Math.max(1, Math.floor(kd + 1.8 * digits**(2/3) * Math.cbrt(kd)));
        const nt = 2*(L+1), np = 2*nt;
        if (nt*np > 30000) throw Error(`This grid has ${nt*np} directions. The browser view stops at 30,000; Julia has no such display limit.`);
        const theta = gauss(nt).x.map(x => Math.PI*(x+1)/2);
        const x = [], y = [], z = [];
        for (const t of theta) for (const p of grid(np)) {
            x.push(Math.sin(t)*Math.cos(p)); y.push(Math.sin(t)*Math.sin(p)); z.push(Math.cos(t));
        }
        return {L, nt, np, theta, x, y, z};
    }
    function transfer(n, target, mode, stencil, family) {
        integer(n, 2); integer(target, n); integer(mode); integer(stencil, 1);
        if (n % 2 || target % 2) throw Error("Source and target counts must be even.");
        if (stencil > n) throw Error("The stencil cannot exceed the source count.");
        if (n*target > 2000000 || stencil > 128) throw Error("This browser illustration is too large; use the Julia example for this case.");
        const source = grid(n), phi = grid(target), values = source.map(p => Math.cos(mode*p));
        let result;
        if (family === "Fourier") {
            const spectrum = source.map((_, q) => {
                const f = q <= n/2 ? q : q-n;
                return [f, values.reduce((s,v,j) => s+v*Math.cos(f*source[j]),0)/n,
                    -values.reduce((s,v,j) => s+v*Math.sin(f*source[j]),0)/n];
            });
            result = phi.map(p => spectrum.reduce((s,[f,a,b]) => s+a*Math.cos(f*p)-b*Math.sin(f*p), 0));
        } else {
            result = phi.map(p => {
                const index = Math.floor(p/tau*n);
                const start = index-Math.floor(stencil/2)+(stencil%2 ? 0 : 1);
                const nodes = Array.from({length:stencil},(_,j) => start+j);
                const u = p/tau*n;
                return nodes.reduce((sum,j) => {
                    let weight = 1;
                    for (const k of nodes) if (j !== k) weight *= (u-k)/(j-k);
                    return sum + weight*values[(j%n+n)%n];
                }, 0);
            });
        }
        const exact = phi.map(p => Math.cos(mode*p));
        return {source, values, phi, exact, result, error: result.map((v,i) => Math.abs(v-exact[i]))};
    }
    function harmonic(z, P) {
        integer(P);
        if (!(z >= 0 && Number.isFinite(z))) throw Error("Use a nonnegative finite angular scale.");
        if (P > 128 || z > 128) throw Error("This browser quadrature view supports degree and scale up to 128.");
        const q = gauss(Math.max(48, 2*P+8, Math.ceil(2*z+16)));
        const a = Array.from({length:P+1},(_,l) => {
            const re = q.x.reduce((s,x,i) => s+q.w[i]*Math.cos(z*x)*legendre(l,x),0);
            const im = -q.x.reduce((s,x,i) => s+q.w[i]*Math.sin(z*x)*legendre(l,x),0);
            return [re*(2*l+1)/2, im*(2*l+1)/2];
        });
        const theta = Array.from({length:301},(_,i) => Math.PI*i/300);
        const exact = theta.map(t => [Math.cos(z*Math.cos(t)), -Math.sin(z*Math.cos(t))]);
        const reconstructed = theta.map(t => a.reduce((s,c,l) => {
            const v = legendre(l,Math.cos(t)); return [s[0]+v*c[0],s[1]+v*c[1]];
        },[0,0]));
        const error = reconstructed.map((v,i) => Math.hypot(v[0]-exact[i][0],v[1]-exact[i][1]));
        return {theta, exact, reconstructed, error, coefficients:a};
    }
    function phase(delta, phi) {
        if (!Number.isFinite(delta)) throw Error("Use a finite displacement.");
        return phi.map(p=>[Math.cos(-tau*delta*Math.cos(p)),Math.sin(-tau*delta*Math.cos(p))]);
    }
    const api = {gauss, sampling, transfer, harmonic, phase};
    if (typeof module !== "undefined") module.exports = api;
    root.MLFMAExplorer = api;
    if (typeof document === "undefined") return;
    const plotlyURL = new URL("generated/plotly.min.js", document.currentScript.src).href;
    const accuracyURL = new URL("generated/accuracy.js", document.currentScript.src).href;
    let accuracyReady;
    function loadAccuracy() {
        if (root.MLFMAAccuracy) return Promise.resolve();
        if (!accuracyReady) accuracyReady = new Promise((resolve,reject) => {
            const script = document.createElement("script"); script.src=accuracyURL;
            script.onload=resolve; script.onerror=()=>reject(Error("Could not load translation-accuracy data."));
            document.head.append(script);
        });
        return accuracyReady;
    }
    let plotlyReady;
    function loadPlotly() {
        if (root.Plotly) return Promise.resolve();
        if (!plotlyReady) plotlyReady = new Promise((resolve,reject) => {
            if (root.requirejs) {
                root.requirejs.config({paths:{"mlfma-plotly":plotlyURL.replace(/\.js$/,"")}});
                root.requirejs(["mlfma-plotly"], p => { root.Plotly=p; resolve(); }, reject);
            } else {
                const script=document.createElement("script"); script.src=plotlyURL;
                script.onload=resolve; script.onerror=()=>reject(Error("Could not load the local plot library."));
                document.head.append(script);
            }
        });
        return plotlyReady;
    }

    function control(form, title, value, options) {
        const label = document.createElement("label"), text = document.createElement("span");
        text.textContent = title; label.append(text);
        const input = document.createElement(options ? "select" : "input");
        if (options) for (const option of options) input.add(new Option(option,option));
        else { input.type = "number"; input.step = Number.isInteger(value) ? "1" : "any"; }
        input.value = value; label.append(input); form.append(label);
        return input;
    }
    function mount(host) {
        if (host.dataset.ready) return;
        host.dataset.ready = "true";
        const form = document.createElement("div"), plot = document.createElement("div"), status = document.createElement("p");
        form.className = "explorer-controls"; plot.className = "explorer-plot";
        plot.setAttribute("role","img"); plot.setAttribute("aria-label",`${host.dataset.kind} interactive figure`);
        status.className = "explorer-status"; status.setAttribute("role", "status");
        host.append(form,plot,status);
        const kind = host.dataset.kind;
        let update;
        const curve = (x,y,name,extra={}) => ({type:"scatter",mode:"lines",x,y,name,...extra});
        const number = input => Number(input.value || NaN);
        function draw(data, xlabel, ylabel, extra={}) {
            const dark = document.documentElement.getAttribute("data-theme") === "dark" || document.documentElement.classList.contains("theme--documenter-dark");
            return Plotly.react(plot, data, {autosize:true, margin:{l:60,r:15,t:15,b:65},
                paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:dark?"#dedede":"#333"},
                colorway:["#389826","#9558b2","#cb3c33","#4063d8"],
                xaxis:{title:{text:xlabel},automargin:true},yaxis:{title:{text:ylabel},automargin:true},
                legend:{orientation:"h",y:-0.22},...extra}, {responsive:true,displaylogo:false});
        }
        if (kind === "sampling") {
            const width=control(form,"Box width / wavelength",0.25), digits=control(form,"Accuracy digits",4);
            update = () => {
                const s=sampling(number(width),number(digits));
                draw([{type:"scatter3d",mode:"markers",x:s.x,y:s.y,z:s.z,marker:{size:2,color:s.z,colorscale:"Viridis"},showlegend:false}],"","",{scene:{aspectmode:"cube",xaxis:{title:"kx"},yaxis:{title:"ky"},zaxis:{title:"kz"}}});
                return `L = ${s.L}; ${s.nt} polar × ${s.np} azimuthal = ${s.x.length} directions.`;
            };
        } else if (kind === "transfer") {
            const family=control(form,"Interpolation","Fourier",["Fourier","Local Lagrange"]);
            const n=control(form,"Source count",16), target=control(form,"Target count",64), mode=control(form,"Mode m",3), p=control(form,"Stencil size",6);
            update = () => {
                p.disabled=family.value==="Fourier";
                const t=transfer(number(n),number(target),number(mode),number(p),family.value);
                draw([curve(t.phi,t.exact,"Exact"),curve(t.phi,t.result,"Interpolated",{line:{dash:"dash"}}),
                    {type:"scatter",mode:"markers",x:t.source,y:t.values,name:"Source samples"}],"φ (radians)","Real field");
                const warning=number(mode)>=number(n)/2 ? " Source mode is at or beyond Nyquist; this is not a resolved-mode accuracy check." : "";
                return `Maximum absolute error: ${Math.max(...t.error).toExponential(2)}.${warning}`;
            };
        } else if (kind === "phase") {
            const delta=control(form,"Center displacement / wavelength",0.25);
            update = () => {
                const phi=grid(361), values=phase(number(delta),phi);
                draw([curve(phi,values.map(v=>v[0]),"Real"),curve(phi,values.map(v=>v[1]),"Imaginary")],"φ (radians)","Radiation phase");
                return "Magnitude is one; the receive phase has the opposite sign.";
            };
        } else if (kind === "harmonics") {
            const z=control(form,"Angular scale z",8), degree=control(form,"Retained degree P",6);
            update = () => {
                const h=harmonic(number(z),number(degree));
                draw([curve(h.theta,h.exact.map(v=>v[0]),"Exact"),curve(h.theta,h.reconstructed.map(v=>v[0]),"Truncated",{line:{dash:"dash"}})],"θ (radians)","Real field");
                return `Maximum complex error: ${Math.max(...h.error).toExponential(2)}; ${(number(degree)+1)**2} scalar spherical coefficients (${number(degree)+1} nonzero m=0 modes here).`;
            };
        } else if (kind === "translation") {
            const r=control(form,"Separation / wavelength","1",["0.5","1","2"]), L=control(form,"Cutoff L","8",["4","8","12"]), view=control(form,"Component","Magnitude",["Magnitude","Phase"]);
            update = () => {
                const f=root.MLFMAFigures.translations[`${r.value}/${L.value}`];
                const y=view.value==="Magnitude"?f.magnitude:f.phase;
                draw([curve(root.MLFMAFigures.phi,y,view.value)],"φ (radians)",view.value==="Magnitude"?"|T|":"arg T (radians)");
                return "Julia-generated translation slice; not a separation-validity test.";
            };
        } else if (kind === "accuracy") {
            plot.classList.add("accuracy-plot");
            const dimension=control(form,"Dimension","3-D",["3-D","2-D"]);
            const family=control(form,"Angular grid","Spherical",["Spherical","Equidistant","Circular"]);
            const offset=control(form,"Local offset / wavelength","0.4",["0.1","0.2","0.4","0.8"]);
            const refinement=control(form,"Angular refinement","1",["1","2","4"]);
            const low=control(form,"First cutoff L",4), high=control(form,"Last cutoff L",17);
            const minimum=control(form,"Minimum kD",0.5), maximum=control(form,"Maximum kD",1000);
            const reference=control(form,"Integrated reference","Hide",["Hide","Show"]);
            update = () => {
                const data=root.MLFMAAccuracy, dim=dimension.value==="2-D"?2:3;
                for(const option of family.options) option.disabled=dim===2 ? option.value!=="Circular" : option.value==="Circular";
                if(dim===2) family.value="Circular"; else if(family.value==="Circular") family.value="Spherical";
                const first=integer(number(low),1), last=integer(number(high),first);
                if(last>data.maxorder) throw Error(`This offline dataset contains cutoffs 1–${data.maxorder}. Generate another range in docs/translation-accuracy.jl for larger orders.`);
                const min=number(minimum), max=number(maximum), a=number(offset);
                if(!(min>=data.kd[0] && max<=1000 && max>min)) throw Error(`Choose a kD interval inside ${data.kd[0]}–1000, with maximum greater than minimum.`);
                const traces=[], actual=[];
                for(let L=first;L<=last;L++) {
                    const entry=data.curves[`${family.value.toLowerCase()}/${refinement.value}/${L}`];
                    const indices=data.kd.map((kd,i)=>kd>=min&&kd<=max?i:-1).filter(i=>i>=0);
                    const x=indices.map(i=>data.kd[i]);
                    const customdata=indices.map(i=>[entry.effective[i],entry.directions]);
                    const color=`hsl(${(L*137.5)%360},55%,48%)`;
                    traces.push(curve(x,indices.map(i=>entry.errors[offset.value][i]),`L = ${L}`,{
                        line:{color,width:1.7},customdata,
                        hovertemplate:"kD=%{x:.3g}<br>relative error=%{y:.3e}<br>effective L=%{customdata[0]}<br>directions=%{customdata[1]}<extra>L = "+L+"</extra>"}));
                    actual.push(...indices.filter(i=>entry.errors[offset.value][i]!==null).map(i=>entry.effective[i]));
                    if(reference.value==="Show") traces.push(curve(x,indices.map(i=>entry.integrated[offset.value][i]),`Integrated L = ${L}`,{
                        line:{color,width:1,dash:"dot"},showlegend:false,hovertemplate:"kD=%{x:.3g}<br>integrated-series error=%{y:.3e}<extra>L = "+L+"</extra>"}));
                }
                if(!actual.length) throw Error("This interval has no samples satisfying D > |x|.");
                const narrow=host.clientWidth<500;
                plot.style.height=(narrow?690:560)+"px";
                draw(traces,"kD","Relative error",{
                    height:narrow?690:560,margin:{l:65,r:15,t:20,b:narrow?210:135},
                    xaxis:{type:"log",title:{text:"kD"},dtick:1,range:[Math.log10(min),Math.log10(max)]},
                    yaxis:{type:"log",title:{text:"Relative error"},dtick:2,exponentformat:"power",showexponent:"all",automargin:true},
                    legend:{orientation:"h",y:narrow?-0.15:-0.2,font:{size:11}},
                    shapes:[{type:"rect",xref:"x",yref:"paper",x0:min,x1:Math.max(min,Math.min(max,tau*a)),y0:0,y1:1,fillcolor:"#8882",line:{width:0},layer:"below"}],
                });
                const guard=dim===2 ? ` Effective cutoff across visible valid points: ${Math.min(...actual)}–${Math.max(...actual)}; the production cancellation guard is active.` : " No cutoff clamping in this 3-D kernel.";
                const ref=reference.value==="Show" ? (dim===3 ? " Dotted reference: 256-bit integrated series." : " Dotted reference: Float64 integrated series, using the effective cutoff.") : "";
                return "Shaded separations fail D > |x| and are omitted. Hover for effective order and direction count."+guard+ref;
            };
        } else if (kind === "symmetry") {
            const angle=control(form,"Rotation","0",["0","90","180","270"]);
            update = () => {
                const a=number(angle)*Math.PI/180, phi=grid(65);
                draw([curve(phi.map(Math.cos),phi.map(Math.sin),"Direction circle"),
                    curve([0,Math.cos(a)],[0,Math.sin(a)],"S R",{line:{width:5}}),
                    curve([0,Math.cos(-a)],[0,Math.sin(-a)],"S⁻¹ k",{line:{width:3,dash:"dash"}})],"x","y",{yaxis:{scaleanchor:"x",range:[-1.2,1.2]},xaxis:{range:[-1.2,1.2]}});
                return "A quarter turn illustrates the inverse convention; the full sphere's supported subgroup can differ.";
            };
        } else throw Error(`Unknown explorer: ${kind}`);
        const render = () => {
            try { status.textContent=update(); status.classList.remove("is-error"); plot.style.visibility="visible"; }
            catch(error) { status.textContent=error.message; status.classList.add("is-error"); plot.style.visibility="hidden"; }
        };
        form.addEventListener("change",render);
        if(kind==="accuracy") window.addEventListener("resize",render);
        const observer = new MutationObserver(render);
        observer.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme","class"]});
        render();
    }
    function boot() {
        if (!document.querySelector(".mlfma-explorer")) return;
        Promise.all([loadPlotly(),document.querySelector('[data-kind="accuracy"]') ? loadAccuracy() : Promise.resolve()]).then(()=>document.querySelectorAll(".mlfma-explorer").forEach(mount))
            .catch(error=>document.querySelectorAll(".mlfma-explorer").forEach(host=>{host.textContent=error.message;}));
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded",boot); else boot();
    if (typeof document$ !== "undefined") document$.subscribe(boot);
})(typeof globalThis !== "undefined" ? globalThis : this);
