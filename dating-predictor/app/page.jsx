"use client";

import { useState } from "react";

const API_URL = "http://localhost:8000";

const OUTCOME_EMOJI = {
  "Mutual Match":       "💞",
  "Relationship Formed":"💍",
  "Instant Match":      "⚡",
  "Date Happened":      "🌹",
  "One-sided Like":     "💔",
  "Ghosted":            "👻",
  "Chat Ignored":       "🔇",
  "No Action":          "😶",
  "Blocked":            "🚫",
  "Catfished":          "🎣",
};

const OUTCOME_COLOR = {
  "Mutual Match":       "#f43f5e",
  "Relationship Formed":"#ec4899",
  "Instant Match":      "#f97316",
  "Date Happened":      "#e11d48",
  "One-sided Like":     "#a855f7",
  "Ghosted":            "#64748b",
  "Chat Ignored":       "#475569",
  "No Action":          "#94a3b8",
  "Blocked":            "#ef4444",
  "Catfished":          "#f59e0b",
};

const Field = ({ label, children }) => (
  <div className="flex flex-col gap-1.5">
    <label className="text-xs font-semibold uppercase tracking-widest text-rose-300">
      {label}
    </label>
    {children}
  </div>
);

const inputClass =
  "bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white " +
  "focus:outline-none focus:border-rose-400 focus:bg-white/10 transition-all " +
  "placeholder-white/30 text-sm";

const selectClass =
  "bg-zinc-900 border border-white/10 rounded-xl px-4 py-2.5 text-white " +
  "focus:outline-none focus:border-rose-400 transition-all text-sm cursor-pointer";

export default function MatchPredictor() {
  const [form, setForm] = useState({
    gender:             "Male",
    location_type:      "Urban",
    income_bracket:     "Middle",
    education_level:    "Bachelor's",
    app_usage_time_min: 60,
    swipe_right_ratio:  0.5,
    profile_pics_count: 5,
    bio_length:         150,
    message_sent_count: 30,
    swipe_time_of_day:  "Evening",
  });

  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const set = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const setNum = (k) => (e) =>
    setForm((f) => ({ ...f, [k]: Number(e.target.value) }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const accent = result ? OUTCOME_COLOR[result.prediction] ?? "#f43f5e" : "#f43f5e";

  return (
    <main className="min-h-screen bg-zinc-950 text-white flex items-center justify-center p-6">
      {/* Ambient glow */}
      <div
        className="pointer-events-none fixed inset-0 opacity-20 transition-all duration-700"
        style={{
          background: `radial-gradient(ellipse 60% 50% at 50% 0%, ${accent}55, transparent)`,
        }}
      />

      <div className="relative w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-5xl mb-3">💘</div>
          <h1 className="text-3xl font-bold tracking-tight">Match Predictor</h1>
          <p className="text-white/40 text-sm mt-1">
            Powered by XGBoost · WIA1006/WID3006
          </p>
        </div>

        {/* Form card */}
        <form
          onSubmit={handleSubmit}
          className="bg-white/5 backdrop-blur border border-white/10 rounded-3xl p-6 space-y-5"
        >
          {/* Row 1 */}
          <div className="grid grid-cols-2 gap-4">
            <Field label="Gender">
              <select className={selectClass} value={form.gender} onChange={set("gender")}>
                {["Male","Female","Non-binary","Other"].map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            </Field>

            <Field label="Location">
              <select className={selectClass} value={form.location_type} onChange={set("location_type")}>
                {["Urban","Suburban","Rural"].map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            </Field>
          </div>

          {/* Row 2 */}
          <div className="grid grid-cols-2 gap-4">
            <Field label="Income bracket">
              <select className={selectClass} value={form.income_bracket} onChange={set("income_bracket")}>
                {["Very Low","Low","Lower-Middle","Middle","Upper-Middle","High","Very High"].map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            </Field>

            <Field label="Education">
              <select className={selectClass} value={form.education_level} onChange={set("education_level")}>
                {["High School","Diploma","Associate's","Bachelor's","MBA","Master's","PhD","Postdoc"].map((o) => (
                  <option key={o}>{o}</option>
                ))}
              </select>
            </Field>
          </div>

          {/* Swipe ratio slider */}
          <Field label={`Swipe-right ratio — ${Math.round(form.swipe_right_ratio * 100)}%`}>
            <input
              type="range" min="0" max="1" step="0.01"
              value={form.swipe_right_ratio}
              onChange={setNum("swipe_right_ratio")}
              className="w-full accent-rose-400 cursor-pointer"
            />
          </Field>

          {/* App usage slider */}
          <Field label={`Daily app usage — ${form.app_usage_time_min} min`}>
            <input
              type="range" min="0" max="300" step="5"
              value={form.app_usage_time_min}
              onChange={setNum("app_usage_time_min")}
              className="w-full accent-rose-400 cursor-pointer"
            />
          </Field>

          {/* Row 3 */}
          <div className="grid grid-cols-3 gap-4">
            <Field label="Profile pics">
              <input type="number" min="1" max="10"
                className={inputClass} value={form.profile_pics_count}
                onChange={setNum("profile_pics_count")} />
            </Field>
            <Field label="Bio length">
              <input type="number" min="0" max="500"
                className={inputClass} value={form.bio_length}
                onChange={setNum("bio_length")} />
            </Field>
            <Field label="Msgs sent">
              <input type="number" min="0"
                className={inputClass} value={form.message_sent_count}
                onChange={setNum("message_sent_count")} />
            </Field>
          </div>

          {/* Active time */}
          <Field label="Most active swipe time">
            <select className={selectClass} value={form.swipe_time_of_day} onChange={set("swipe_time_of_day")}>
              {["Early Morning","Morning","Afternoon","Evening","Night","After Midnight"].map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </Field>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-2xl font-bold text-sm tracking-wide transition-all duration-200
                       bg-rose-500 hover:bg-rose-400 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Predicting…" : "Predict My Match Outcome ✨"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-2xl p-4 text-red-300 text-sm text-center">
            ⚠️ {error} — make sure the FastAPI server is running on port 8000.
          </div>
        )}

        {/* Result card */}
        {result && (
          <div
            className="mt-5 rounded-3xl p-6 text-center border transition-all duration-500"
            style={{
              background: `${accent}15`,
              borderColor: `${accent}40`,
            }}
          >
            <div className="text-5xl mb-2">
              {OUTCOME_EMOJI[result.prediction] ?? "🎯"}
            </div>
            <div className="text-2xl font-bold" style={{ color: accent }}>
              {result.prediction}
            </div>
            <div className="text-white/40 text-sm mt-1">
              {result.confidence}% confidence
            </div>

            {/* Top-3 bar */}
            <div className="mt-5 space-y-2 text-left">
              <p className="text-xs text-white/30 uppercase tracking-widest mb-3">
                Top predictions
              </p>
              {result.top3.map((item) => (
                <div key={item.outcome} className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-36 truncate">
                    {item.outcome}
                  </span>
                  <div className="flex-1 bg-white/5 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all duration-700"
                      style={{
                        width:      `${item.probability}%`,
                        background: OUTCOME_COLOR[item.outcome] ?? accent,
                      }}
                    />
                  </div>
                  <span className="text-xs text-white/40 w-10 text-right">
                    {item.probability}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <p className="text-center text-white/20 text-xs mt-6">
          WIA1006/WID3006 Machine Learning · Group Assignment 2025/2026
        </p>
      </div>
    </main>
  );
}