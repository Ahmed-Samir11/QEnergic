import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Polygon, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import { cellTypes } from '../data/cellTypes';
import africaCities from '../data/africa_cities.json';

// Leaflet marker icons fix
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png',
});

function MarkerHandler({ markers, setMarkers, popCenter, setPopCenter, popMode }) {
  useMapEvents({
    click(e) {
      if (popMode) {
        setPopCenter({ lat: e.latlng.lat, lng: e.latlng.lng });
      } else if (markers.length < 4) {
        setMarkers(prev => [...prev, { lat: e.latlng.lat, lng: e.latlng.lng }]);
      }
    }
  });
  return null;
}

export default function MapOptimizer() {
  const [markers, setMarkers] = useState([]); // Only 4 allowed
  const [budget, setBudget] = useState(10000); // New: budget input
  const [gridArea, setGridArea] = useState(1); // Still needed for area check
  const [grids, setGrids] = useState([]); // Optimized grid centers
  const [cellTypeIdx, setCellTypeIdx] = useState(0);
  const [cellSizeIdx, setCellSizeIdx] = useState(0);
  const [countryIdx, setCountryIdx] = useState(0);
  const [popCenter, setPopCenter] = useState(null);
  const [popMode, setPopMode] = useState(false);
  const [algo, setAlgo] = useState('classical'); // 'classical' or 'quantum'
  const [optTime, setOptTime] = useState(null);

  const selectedOption = cellTypes[cellTypeIdx].options[cellSizeIdx];
  const selectedCountry = africaCities[countryIdx];

  // Remove marker on right-click
  const removeMarker = idx => {
    setMarkers(prev => prev.filter((_, i) => i !== idx));
  };

  // Remove pop center on right-click
  const removePopCenter = () => setPopCenter(null);

  // Optimize grid placement
  const solve = async (algorithm = 'classical') => {
    if (markers.length !== 4) {
      alert('Please place exactly 4 markers to define the area.');
      return;
    }
    if (gridArea <= 0) {
      alert('Grid area must be positive.');
      return;
    }
    if (!popCenter) {
      alert('Please mark the center of population on the map.');
      return;
    }
    setAlgo(algorithm);
    setOptTime(null);
    try {
      const t0 = performance.now();
      let newGrids = [];
      if (algorithm === 'classical') {
        const res = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            polygon: markers,
            budget: Number(budget),
            gridArea: Number(gridArea),
            cellSize: selectedOption,
            country: selectedCountry.name,
            popCenter,
          })
        });
        const data = await res.json();
        newGrids = data.grids || [];
      } else if (algorithm === 'quantum') {
        // Call classical API to get QUBO and candidates, then call quantum_optimize.py
        const res = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            polygon: markers,
            budget: Number(budget),
            gridArea: Number(gridArea),
            cellSize: selectedOption,
            country: selectedCountry.name,
            popCenter,
            quboOnly: true,
          })
        });
        const data = await res.json();
        if (!data.Q || !data.candidates) throw new Error('QUBO not returned');
        // Save Q and candidates to temp files and call quantum_optimize.py
        const resp = await fetch('/api/quantum_optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ Q: data.Q, k: gridCount, candidates: data.candidates })
        });
        const qres = await resp.json();
        newGrids = (qres.selected_indices || []).map(i => data.candidates[i]);
      } else if (algorithm === 'gurobi') {
        const resp = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ polygon: markers, budget: Number(budget), gridArea: Number(gridArea), cellSize: selectedOption, country: selectedCountry, popCenter, algo: 'gurobi' })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        newGrids = data.grids;
        setOptTime(data.time_ms);
      } else if (algorithm === 'nar') {
        const res = await fetch('/api/optimize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            polygon: markers,
            budget: Number(budget),
            gridArea: Number(gridArea),
            cellSize: selectedOption,
            country: selectedCountry.name,
            popCenter,
            algo: 'nar'
          })
        });
        const data = await res.json();
        newGrids = data.grids || [];
      }
      setGrids(newGrids);
      setOptTime(performance.now() - t0);
    } catch (error) {
      alert('Optimization failed.');
    }
  };

  return (
    <div className="w-full max-w-4xl">
      <div className="mb-4">
        <label className="mr-2">Country:</label>
        <select value={countryIdx} onChange={e => setCountryIdx(+e.target.value)} className="border p-1 rounded mr-4">
          {africaCities.map((city, i) => <option value={i} key={city.name}>{city.name}</option>)}
        </select>
        <button onClick={() => setPopMode(m => !m)} className={`ml-2 px-2 py-1 rounded ${popMode ? 'bg-green-600 text-white' : 'bg-gray-200'}`}>{popMode ? 'Marking: Click Map' : 'Mark Center of Population'}</button>
        {popCenter && <button onClick={removePopCenter} className="ml-2 px-2 py-1 rounded bg-red-200">Remove Pop Center</button>}
      </div>
      <MapContainer center={[selectedCountry.lat, selectedCountry.lng]} zoom={5} style={{ height: '500px', marginBottom: '1rem' }}>
        <MarkerHandler markers={markers} setMarkers={setMarkers} popCenter={popCenter} setPopCenter={setPopCenter} popMode={popMode} />
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {markers.map((m, i) => (
          <Marker key={i} position={[m.lat, m.lng]} eventHandlers={{
            contextmenu: () => removeMarker(i)
          }}>
            <Popup>Marker {i + 1} (Right-click to remove)</Popup>
          </Marker>
        ))}
        {markers.length === 4 && (
          <Polygon positions={markers.map(m => [m.lat, m.lng])} pathOptions={{ color: 'blue' }} />
        )}
        {popCenter && (
          <Marker position={[popCenter.lat, popCenter.lng]} icon={L.divIcon({className: 'pop-center-marker', html: '<div style="background:orange;width:18px;height:18px;border-radius:50%;border:2px solid #fff"></div>'})}>
            <Popup>Population Center (Right-click to remove)</Popup>
          </Marker>
        )}
        {grids.map((g, i) => (
          <Marker key={`grid-${i}`} position={[g.lat, g.lng]} icon={L.divIcon({className: 'grid-marker', html: '<div style="background:red;width:16px;height:16px;border-radius:50%"></div>'})}>
            <Popup>Grid {i + 1}</Popup>
          </Marker>
        ))}
      </MapContainer>
      <div className="flex items-center mb-4">
        <div className="mb-2">
          <label>Budget ($): </label>
          <input type="number" min={0} value={budget} onChange={e => setBudget(e.target.value)} className="border px-2 py-1 w-32" />
        </div>
        <div className="mb-2">
          <label>Area per Grid (m²): </label>
          <input type="number" value={gridArea} min={0.01} step={0.01} onChange={e => setGridArea(+e.target.value)} className="border px-2 py-1 w-32" />
        </div>
        <button onClick={() => solve('nar')} className="ml-4 bg-red-600 text-white px-3 py-1 rounded">Greedy (NAR)</button>
        <button onClick={() => solve('classical')} className="ml-2 bg-blue-600 text-white px-3 py-1 rounded">Simulated Annealing</button>
        <button onClick={() => solve('gurobi')} className="ml-2 bg-green-700 text-white px-3 py-1 rounded">Gurobi</button>
      </div>
      {optTime && (
        <div className="mb-2 text-green-700">Optimization time: {(optTime/1000).toFixed(2)} seconds ({algo})</div>
      )}
      {grids.length > 0 && (
        <div className="mt-4 p-4 bg-white border rounded">
          <h2 className="text-lg font-semibold mb-2">Grid Centers</h2>
          <ul className="list-disc pl-5">
            {grids.map((g, i) => (
              <li key={i}>Lat: {g.lat.toFixed(4)}, Lng: {g.lng.toFixed(4)}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="mt-2 text-sm text-gray-600">Click to add up to 4 markers (right-click to remove). These define the area of interest. Use the button to mark the population center.</div>
      <div className="mt-4">
        <label className="mr-2">Cell Type:</label>
        <select value={cellTypeIdx} onChange={e => { setCellTypeIdx(+e.target.value); setCellSizeIdx(0); }} className="border p-1 rounded mr-4">
          {cellTypes.map((ct, i) => <option value={i} key={ct.type}>{ct.type}</option>)}
        </select>
        <label className="mr-2">Cell Size:</label>
        <select value={cellSizeIdx} onChange={e => setCellSizeIdx(+e.target.value)} className="border p-1 rounded">
          {cellTypes[cellTypeIdx].options.map((opt, i) => <option value={i} key={opt.label}>{opt.label}</option>)}
        </select>
      </div>
    </div>
  );
}